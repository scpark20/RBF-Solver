CKPT_PATH="pretrained/edm-cifar10-32x32-uncond-vp.pkl"

for steps in 5 6 8 10 12 15 20 25 30 35 40; do

python sample.py --sample_folder="dpm_solver++_"$steps --ckp_path=$CKPT_PATH --method=dpm_solver++ --steps=$steps --skip_type=logSNR

python sample.py --sample_folder="uni_pc_bh1_"$steps --unipc_variant=bh1 --ckp_path=$CKPT_PATH --method=uni_pc --steps=$steps --skip_type=logSNR

python sample.py --sample_folder="uni_pc_bh2_"$steps --unipc_variant=bh2 --ckp_path=$CKPT_PATH --method=uni_pc --steps=$steps --skip_type=logSNR

done