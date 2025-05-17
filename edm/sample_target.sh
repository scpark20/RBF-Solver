CKPT_PATH="pretrained/edm-cifar10-32x32-uncond-vp.pkl"
steps=200
python sample_target.py --sample_folder="uni_pc_bh2_"$steps --unipc_variant=bh2 --ckp_path=$CKPT_PATH --method=uni_pc --steps=$steps --skip_type=logSNR