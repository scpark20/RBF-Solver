CKPT_PATH="pretrained/edm-cifar10-32x32-uncond-vp.pkl"
SHAPE_DIR="shape_dir"
PAIR_PT='samples/edm-cifar10-32x32-uncond-vp/uni_pc_bh2_200/samples_0.pt'

for order in 3 4; do
for steps in 5 6 8 10 12 15 20 25 30 35 40; do
python shape_optim.py --shape_dir=$SHAPE_DIR --ckp_path=$CKPT_PATH --pair_pt=$PAIR_PT --method=rbf_solver --order=$order --steps=$steps --skip_type=logSNR
done
done

