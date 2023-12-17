export PYTHONDONTWRITEBYTECODE=1
python  main.py  data.data_path=/fs/cfar-projects/frequency_stuff/vid_inr/frames/jockey_1080/ \
logging.checkpoint.logdir=output/temp_jockey/ \
trainer.num_iters=2000 \
trainer.num_iters_first=16000 \
network=siren_nerv_optim \
network.layer_size=512 \
network.num_layers=3 \
trainer.group_size=3 trainer.batch_size=1 \
data.patch_shape=[32,32] \
network.w0=30 \
network.w0_initial=30 \
trainer.lr=5e-4 data.max_frames=9 \
trainer.reset_best=True \
trainer.precision=32 \
trainer.strategy='ddp' \
common.seed=4341 data.data_format="patch_first" trainer.devices=[0] \
trainer.num_workers=4 \
trainer.losses.loss_list=['mse','entropy_reg'] \
network.decoder_cfg.no_shift=True \
network.prob_num_layers=1 \
network.init_mode='prev'

