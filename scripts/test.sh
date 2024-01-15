export PYTHONDONTWRITEBYTECODE=1
python  main.py  data.data_path=/fs/cfar-projects/frequency_stuff/vid_inr/frames/jockey_1080/ \
logging.checkpoint.logdir=output/temp_jockey/ \
trainer.num_iters=2 \
trainer.num_iters_first=8 \
network=siren_nerv \
network.layer_size=512 \
network.num_layers=3 \
trainer.group_size=3 trainer.batch_size=1 \
data.patch_shape=[32,32] \
network.w0=30 \
network.w0_initial=30 \
trainer.lr=5e-4 data.max_frames=99 \
trainer.reset_best=True \
common.seed=4341 trainer.devices=[0] \
trainer.num_workers=4 \
trainer.losses.loss_list=['mse','entropy_reg'] \
network.decoder_cfg.no_shift=True \
network.prob_num_layers=1 \
network.init_mode='prev' trainer.eval_only=False \
logging.checkpoint.skip_save_model=True


