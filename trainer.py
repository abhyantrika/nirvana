from pydoc import locate

import torch
import torchac 
import wandb
import tqdm
import copy
import time
import itertools
import glob,os,lzma
from einops import rearrange
from omegaconf import DictConfig,OmegaConf

from models import layers
from utils import helper,metric,state_tools,dist_sampler,data_process
import dataloader

class Trainer():
    def __init__(self,cfg,local_rank=0,world_size=1) -> None:
        
        self.cfg = cfg
        self.local_rank = local_rank
        self.world_size = world_size
        
        #load base model
        self.model_name = self.cfg.network.model_name
        self.model_class = locate('models.'+ self.model_name + '.'+ self.model_name)
        self.model = self.model_class(self.cfg)
        
        self.load_loss()
        self.load_optimizer()
        self.load_loggers()
        self.load_dataloader()
        
        self.save_root = self.cfg.logging.checkpoint.logdir

        self.device = torch.device("cuda:{}".format(self.local_rank))
        torch.cuda.set_device(self.device)
        self.model.to(self.device)
        
        for k in self.model.prob_models.keys():
            self.model.prob_models[k].to(self.device)

        #variables. 
        self.coordinates = None
        self.proc = None


    def load_loggers(self):

        if self.cfg.trainer.resume:
            wandb_id = self.cfg.logging.wandb.id
            resume = 'must'
        else:
            wandb_id = wandb.util.generate_id()
            resume = None

        if self.cfg.logging.wandb.enabled:
            self.wandb_config = self.cfg.logging.wandb
            wandb_config = helper.flatten_dictconfig(self.cfg)
            self.logger = wandb.init(project=self.wandb_config.project,config=wandb_config,group=self.wandb_config.group,\
                                    save_dir=self.wandb_config.wandb_dir,entity=self.wandb_config.entity,id=wandb_id,resume=resume)
        else:
            self.logger = None


    def save_config(self):
        OmegaConf.save(self.cfg, self.cfg.logging.checkpoint.logdir+'/exp_config.yaml')


    def load_loss(self):
        loss_cfg = self.cfg.trainer.losses
        loss_list = loss_cfg.loss_list
        self.loss_functions = {}
        self.loss_weights = {}

        self.compress = True if  'entropy_reg' in  self.cfg.trainer.losses.loss_list else False

        for idx,loss in enumerate(loss_list):
            loss_name = loss.lower()
            loss_class = locate('losses.'+loss_name)
            loss_func = loss_class(self.cfg)
            self.loss_weights[loss_name] = loss_cfg.loss_weights[idx]
            self.loss_functions[loss_name] = loss_func

            print('Using loss function : ',loss_name,' with weight: ',self.loss_weights[loss_name])  

    def apply_loss(self,outputs,batch,latents=None):
        loss_dict = {}
        num_bits = -1
        for loss_name,loss_func in self.loss_functions.items():
            if loss_name == 'entropy_reg':
                loss_val,num_bits = loss_func(latents,self.model.prob_models,\
                                            self.cfg.network.single_prob_model,\
                                            self.loss_weights[loss_name])
                loss_dict[loss_name] = loss_val
                
            else:
                loss_val = loss_func(outputs,batch['features'].squeeze()) * self.loss_weights[loss_name]
                loss_dict[loss_name] = loss_val
                num_bits = -1

        return loss_dict,num_bits

    def model_stats(self,frame):
        self.model_size = helper.model_size_in_bits(self.model) / 8000.
        print(f'Model size: {self.model_size:.1f}kB')
        print(f'Uncompressed Model size: {helper.uncompressed_model_size_in_bits(self.model)/8000:.1f}kB')
        self.fp_bpp = helper.bpp(model=self.model, image=frame)
        print(f'Full precision bpp: {self.fp_bpp:.2f}')
        self.pytorch_total_params = sum(p.numel() for p in self.model.parameters()) /1e6
        print(f'Pytorch total params: {self.pytorch_total_params} M')

    def load_optimizer(self):
        
        conf_decoder = self.cfg.network.decoder_cfg
        conf_opt = self.cfg.trainer.optimizer

        param_groups = []
        uncompressed_params = 0
        for n,p in self.model.named_parameters():
            if not (n.endswith('linear.weight') or 'dft' in n or 'pos_encoding' in n):
                uncompressed_params += p.numel()
            if n.endswith('0.linear.weight') and not conf_decoder['compress_first']:
                uncompressed_params += p.numel()
            if n.endswith('linear.weight') or ('decoder' in n and conf_decoder['decode_matrix']!='dft_fixed'):
                param_groups += [{'params':[p],'lr':conf_opt['lr']*10,'name':n} if '.0.' in n or 'decoder' in n \
                            else {'params':[p],'lr':conf_opt['lr']*10,'name':n, 'weight_decay':conf_opt['weight_decay']}]
        print(f'Pytorch uncompressed params:{uncompressed_params/10**3}K')
        param_groups += [
                        {
                            'params':[p for n,p in self.model.named_parameters() 
                                        if not ('decoder' in n or n.endswith('linear.weight')
                                            and not ('scale' in n and conf_decoder['decode_matrix']=='dft_fixed'))], 
                            'lr': conf_opt['lr'],
                            'name':'params'
                            }
                         ]
        self.optimizer = torch.optim.Adam(param_groups)

        if self.compress:
            prob_model_parameters = []
            for prob_model in self.model.prob_models.values():
                prob_model_parameters = itertools.chain(prob_model_parameters,prob_model.parameters())
            self.prob_optimizer = torch.optim.Adam(prob_model_parameters, lr = conf_opt['prob_lr'])

    def load_dataloader(self):
        self.dataset = dataloader.VideoFramesDataset(self.cfg)

        if self.cfg.trainer.distributed:
            sampler = dist_sampler.DistributedSampler(self.dataset,shuffle=False)
        else:
            sampler = None

        self.dataloader = torch.utils.data.DataLoader(self.dataset,batch_size=self.cfg.trainer.batch_size,\
                                                      shuffle=False,num_workers=self.cfg.trainer.num_workers,\
                                                        sampler=sampler,collate_fn=dataloader.custom_collate_fn)

    def load_coordinates(self,features_shape):
        self.coordinates = self.proc.get_coordinates(data_shape=features_shape,\
                                            patch_shape=self.cfg.data.patch_shape,\
                                            normalize_range=self.cfg.data.coord_normalize_range)        

    def save_artefacts(self,best_model_state,best_opt_state,best_prediction,save_dir,batch):

        if not self.cfg.logging.checkpoint.skip_save_model:
            helper.save_pickle(best_model_state,filename=save_dir+'/best_model.ckpt',compressed=True)
            helper.save_pickle(best_opt_state,filename=save_dir+'/best_opt.ckpt',compressed=True)

        if self.compress:
            helper.save_pickle(self.byte_stream,filename=save_dir+'/compressed_state.pkl',compressed=True)

        if not self.cfg.logging.checkpoint.skip_save:
            frame_ids = batch['frame_ids'].squeeze().tolist()

            for i in range(len(frame_ids)):
                name = str(frame_ids[i])
                helper.save_tensor_img(best_prediction[i],filename=save_dir+'/pred_'+name+'.png')


    def compute_ac_bytes(self,weights, use_diff_ac, use_prob_model, fit_linear):
        ac_bytes = 0
        overhead = []
        with torch.no_grad():
            for group_name in weights:
                if group_name == 'no_compress':
                    continue
                weight = torch.round(weights[group_name])
                if use_diff_ac:
                    assert not use_prob_model, "Not implemented prob model for diff"
                    if fit_linear:
                        X = torch.round(self.previous_latents[group_name])
                        block_size = self.model.weight_decoders[group_name].block_size
                        for m in self.model.modules():
                            if isinstance(m, layers.Linear):
                                if self.model.groups[m.name] == group_name:
                                    out_features = m.out_features
                                    break
                        X = rearrange(X, '(b c) (b1 c1) -> (b b1) (c c1)', b1=block_size[0], 
                                            c1=block_size[1], b=out_features//block_size[0])
                        X = torch.cat((X.unsqueeze(-1),torch.ones_like(X).unsqueeze(-1)),dim=-1)
                        Y = weights[group_name]
                        Y = rearrange(Y, '(b c) (b1 c1) -> (b b1) (c c1)', b1=block_size[0], 
                                            c1=block_size[1], b=out_features//block_size[0]).unsqueeze(-1)
                        try:
                            out = torch.linalg.inv(torch.matmul(X.permute(0,2,1),X))
                            out = torch.matmul(torch.matmul(out,X.permute(0,2,1)),Y)
                            overhead += [out.detach().cpu()]
                            pred_Y = torch.matmul(X,out)
                            weight = torch.round(Y-pred_Y).reshape(weights[group_name].size())
                        except Exception as e:
                            weight = torch.round(weights[group_name]) - torch.round(self.previous_latents[group_name])
                    else:
                        weight = weight - torch.round(self.previous_latents[group_name])
                for dim in range(weight.size(1)):
                    weight_pos = weight[:,dim] - torch.min(weight[:,dim])
                    unique_vals, counts = torch.unique(weight[:,dim], return_counts = True)
                    if use_prob_model:
                        unique_vals = torch.cat((torch.Tensor([unique_vals.min()-0.5]).to(unique_vals),\
                                                (unique_vals[:-1]+unique_vals[1:])/2,
                                                torch.Tensor([unique_vals.max()+0.5]).to(unique_vals)))
                        cdf = self.model.prob_models[group_name](unique_vals,single_channel=dim)
                        cdf = cdf.detach().cpu().unsqueeze(0).repeat(weight.size(0),1)
                    else:
                        cdf = torch.cumsum(counts/counts.sum(),dim=0).detach().cpu()
                        cdf = torch.cat((torch.Tensor([0.0]),cdf))
                        cdf = cdf/cdf[-1]
                        cdf = cdf.unsqueeze(0).repeat(weight.size(0),1)
                    weight_pos = weight_pos.long()
                    unique_vals = torch.unique(weight_pos)
                    mapping = torch.zeros((weight_pos.max().item()+1))
                    mapping[unique_vals] = torch.arange(unique_vals.size(0)).to(mapping)
                    weight_pos = mapping[weight_pos.cpu()]
                    byte_stream = torchac.encode_float_cdf(cdf.clamp(min=0.0,max=1.0).detach().cpu(), weight_pos.detach().cpu().to(torch.int16), \
                                                    check_input_bounds=True)
                    ac_bytes += len(byte_stream)
        return ac_bytes+sum([torch.finfo(t.dtype).bits/8*t.numel() for t in overhead]),byte_stream

    def train(self):
        encoding_time = 0
        for idx,batch in enumerate(self.dataloader):
            #print(batch['group_id'],batch['frame_ids'])
            group_id = batch['group_id'].item()
            features_shape = batch['features_shape']
            input_image_shape = batch['og_data_shape'].tolist()

            save_dir = self.save_root +'/group_'+ str(group_id) +'/'
            helper.make_dir(save_dir)
            
            ckpt_path = helper.find_ckpt(save_dir)
            if ckpt_path is not None:
                print("Skipping: ",idx)
                continue

            if self.coordinates is None:
                self.proc = data_process.DataProcessor(self.cfg.data)
                self.load_coordinates(features_shape)
                self.coordinates = self.coordinates.to(self.device)

            if idx == 0:
                iterations = self.cfg.trainer.num_iters_first if self.cfg.trainer.num_iters_first is not None else self.cfg.trainer.num_iters
                self.cfg.data.image_shape = input_image_shape
                self.cfg.data.features_shape = features_shape.tolist()
                self.cfg.data.num_frames = self.dataset.num_frames
                self.save_config()

            if idx > 0:
                prev_dir_path = self.save_root +'/group_'+ str(group_id-1) +'/'
                prev_ckpt_path = prev_dir_path + '/best_model.ckpt'
                iterations = self.cfg.trainer.num_iters
                #initialize model with previous model
                if os.path.exists(prev_ckpt_path):
                    prev_model_state = helper.load_pickle(prev_ckpt_path,compressed=True)
                else:
                    prev_model_state = self.prev_best_model_state #best_model_state
                self.model.load_state_dict(prev_model_state)
                

            batch = {key: value.to(self.device) for key, value in batch.items() if type(value) is torch.Tensor}
            self.best_model_state,self.best_opt_state,net_time,best_prediction = self.train_loop(batch,iterations = iterations)

            if self.compress:            
                weights = self.model.get_latents()
                ac_bytes_diff_emp,byte_stream_diff_emp = self.compute_ac_bytes(weights, True and idx>0, False, False)

                if self.cfg.network.use_diff_ac:
                    ac_bytes_emp,byte_stream_emp = self.compute_ac_bytes(weights, False, False, False)
                    ac_bytes_prob,byte_stream_prob = self.compute_ac_bytes(weights, False, True, False)
                    ac_bytes = min(min(ac_bytes_emp, ac_bytes_prob), ac_bytes_diff_emp)
                    self.byte_stream = byte_stream_diff_emp if ac_bytes == ac_bytes_diff_emp else \
                                    byte_stream_emp if ac_bytes == ac_bytes_emp else byte_stream_prob
                    print(f'AC Kbytes Diff: {ac_bytes_diff_emp/1000}, Framewise: {ac_bytes_emp/1000}, Framewise Prob Model: {ac_bytes_prob/1000}')
                else:
                    ac_bytes = ac_bytes_diff_emp                
                    self.byte_stream = byte_stream_diff_emp

                with torch.no_grad():
                    if self.cfg.network.init_mode == 'prev' or idx == 0:
                        self.previous_latents = copy.deepcopy(weights)
                        self.prev_best_model_state = copy.deepcopy(self.best_model_state)

                    elif 'res' in self.cfg.network.init_mode:
                        for group in weights:
                            self.previous_latents[group] = weights[group]+self.cfg.network.res_coeff * (weights[group]-self.previous_latents[group])
                            if 'norm' in self.cfg.network.init_mode:
                                self.previous_latents[group] = self.previous_latents[group]*torch.norm(weights[group])/torch.norm(self.previous_latents[group])

                        for k in self.best_model_state.keys():
                            self.prev_best_model_state[k] = self.best_model_state[k] + self.cfg.network.res_coeff*(self.best_model_state[k]-self.prev_best_model_state[k])
                            if 'norm' in self.cfg.network.init_mode:
                                self.prev_best_model_state[k] = self.prev_best_model_state[k]*torch.norm(self.best_model_state[k])/torch.norm(self.prev_best_model_state[k])
            
            patch_shape = self.cfg.data.patch_shape
            H,W = features_shape[-2:]            
            best_prediction = self.proc.process_outputs(best_prediction,input_image_shape,features_shape,patch_shape=patch_shape,group_size=self.cfg.trainer.group_size)
            self.save_artefacts(self.best_model_state,self.best_opt_state,best_prediction,save_dir,batch)
            encoding_time += net_time
            

    def train_loop(self,batch,iterations):
        self.model.train()
        features = batch['features'].squeeze()
        iteration = tqdm.tqdm(range(iterations),desc='Training',position=0,leave=True)
        best_psnr = 0        
        net_time = 0
        best_model_state = copy.deepcopy(self.model.state_dict())

        for i in iteration:

            if i == 0:
                for g in self.optimizer.param_groups:
                    if g['name'].endswith('linear.weight') and self.compress:                        
                        g['lr'] = self.cfg.trainer.optimizer.lr * 100
                    elif batch['group_id']>=1 and self.cfg.network.encode_residuals:
                        g['lr'] = self.cfg.trainer.optimizer.lr

            start_iter = time.time()
            self.optimizer.zero_grad()

            if self.compress:
                self.prob_optimizer.zero_grad()
                latents = self.model.get_latents()
                conf_decoder = self.cfg.network.decoder_cfg
                with torch.no_grad():
                    if conf_decoder.decode_norm !='none' and i%10==0:
                        for group_name in sorted(self.model.unique_groups):
                            if group_name == 'no_compress':
                                continue
                            cur_weights = torch.round(latents[group_name])
                            if conf_decoder.decode_norm == 'min_max':
                                decoder = self.model.weight_decoders[group_name]
                                decoder.div = torch.max(torch.abs(cur_weights.min(dim=0,keepdim=True)[0]),\
                                                        torch.abs(cur_weights.max(dim=0,keepdim=True)[0]))
                            elif conf_decoder.decode_norm == 'mean_std':
                                decoder = self.model.weight_decoders[group_name]
                                decoder.div = cur_weights.std(dim=0,keepdim=True)
                            decoder.div[decoder.div==0] += 1

            outputs = self.model(self.coordinates)
            
            if self.compress:
                loss_dict,num_bits = self.apply_loss(outputs,batch,latents)
                prob_kbytes = num_bits/8000.
            else:
                loss_dict,loss = self.apply_loss(outputs,batch)
                prob_kbytes = -1
            
            loss = sum([loss_dict[k] for k in loss_dict.keys()])
            
            loss.backward()
            self.optimizer.step()

            if self.compress:
                self.prob_optimizer.step()

            psnr = helper.get_clamped_psnr(features,outputs)

            if psnr > best_psnr:
                best_psnr = psnr
                best_prediction = outputs
                for k, v in self.model.state_dict().items():
                    if k not in best_model_state:
                        best_model_state[k] = v
                    best_model_state[k].copy_(v)   
                best_opt_state = copy.deepcopy(self.optimizer.state_dict())

            net_time += (time.time()-start_iter)

            log_dict = {'loss':loss.item(),'psnr':psnr.item(),'best_psnr':best_psnr.item(),'net_time':net_time,'prob_kbytes':prob_kbytes}
            iteration.set_postfix(**log_dict)

        print("Total encoding time: ",net_time)
        return best_model_state,best_opt_state,net_time,best_prediction
        
    def infer(self):
        
        checkpoint_dir = self.cfg.logging.checkpoint.logdir
        all_model_files = glob.glob(checkpoint_dir + '/*/*.ckpt')
        all_model_files = [x for x in all_model_files if 'opt' not in x]
        all_model_files = sorted(all_model_files)

        if all_model_files == []:
            print("No models found. Exiting.")
            return

        model_state_ops = {}
        for idx,x in enumerate(all_model_files):
            state = helper.load_pickle(x,compressed=True)
            model_state_ops[idx] = state

        

        if self.coordinates is None:
            features_shape = self.cfg.data.features_shape
            self.proc = data_process.DataProcessor(self.cfg.data)
            self.load_coordinates(features_shape)
            self.coordinates = self.coordinates.to(self.device)
        
        total_time = 0
        for group_idx,state in model_state_ops.items():
            torch.cuda.synchronize()
            start = time.time()
            self.model.load_state_dict(state)
            self.model.eval()
            with torch.no_grad():

                ## TODO : do this operation before. 
                if self.compress:
                    latents = self.model.get_latents()
                    conf_decoder = self.cfg.network.decoder_cfg
                    with torch.no_grad():
                        if conf_decoder.decode_norm !='none':
                            for group_name in sorted(self.model.unique_groups):
                                if group_name == 'no_compress':
                                    continue
                                cur_weights = torch.round(latents[group_name])
                                if conf_decoder.decode_norm == 'min_max':
                                    decoder = self.model.weight_decoders[group_name]
                                    decoder.div = torch.max(torch.abs(cur_weights.min(dim=0,keepdim=True)[0]),\
                                                            torch.abs(cur_weights.max(dim=0,keepdim=True)[0]))
                                elif conf_decoder.decode_norm == 'mean_std':
                                    decoder = self.model.weight_decoders[group_name]
                                    decoder.div = cur_weights.std(dim=0,keepdim=True)
                                decoder.div[decoder.div==0] += 1

                outputs = self.model(self.coordinates)
            end = time.time()
            torch.cuda.synchronize()
            total_time += (end-start)

        print("Total time: ",total_time)
        print('FPS: ',self.cfg.data.num_frames/total_time)


