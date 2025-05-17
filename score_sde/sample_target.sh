CKPT_PATH="checkpoints/cifar10_ddpmpp_deep_continuous/checkpoint_8.pth"
CONFIG="configs/vp/cifar10_ddpmpp_deep_continuous.py"
steps=200
EPS="1e-4"

python sample_target.py --config=$CONFIG --ckp_path=$CKPT_PATH --sample_folder="UniPC_bh1_"$steps --config.sampling.method=uni_pc --config.sampling.steps=$steps --config.sampling.variant=bh1 --config.sampling.eps=$EPS