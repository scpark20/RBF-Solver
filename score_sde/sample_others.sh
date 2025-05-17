CKPT_PATH="checkpoints/cifar10_ddpmpp_deep_continuous/checkpoint_8.pth"
CONFIG="configs/vp/cifar10_ddpmpp_deep_continuous.py"
SHAPE_DIR="shape_dir"
for steps in 5 6 8 10 12 15 20 25 30 35 40; do

if [ $steps -le 10 ]; then
    EPS="1e-3"
else
    EPS="1e-4"
fi

python sample.py --config=$CONFIG --ckp_path=$CKPT_PATH --sample_folder="DPM-Solver++_"$steps --config.sampling.method=dpm_solver --config.sampling.steps=$steps --config.sampling.eps=$EPS
python sample.py --config=$CONFIG --ckp_path=$CKPT_PATH --sample_folder="UniPC_bh1_"$steps --config.sampling.method=uni_pc --config.sampling.steps=$steps --config.sampling.variant=bh1 --config.sampling.eps=$EPS
done