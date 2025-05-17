CKPT_PATH="pretrained/edm-cifar10-32x32-uncond-vp.pkl"
SHAPE_DIR="shape_dir"

for order in 3 4; do
for steps in 5 6 8 10 12 15 20 25 30 35 40; do
python sample.py --sample_folder="rbf_solver_"$steps --ckp_path=$CKPT_PATH --method=rbf_solver -order=$order --steps=$steps --skip_type=logSNR
done
done