DEVICES='0'

data="imagenet64"
sampleMethod='rbf_solver'
type="data_prediction"
DIS="logSNR"
method="multistep"

for order in 3 4; do
  for steps in 5 6 8 10 12 15 20 25 30 35 40; do
    workdir="experiments/${data}/${sampleMethod}_order${order}_${steps}"
    CUDA_VISIBLE_DEVICES=$DEVICES python main.py \
      --config "${data}.yml" \
      --exp="$workdir" \
      --sample --fid \
      --shape_dir "shape_dir" \
      --timesteps=$steps --eta 0 --ni \
      --skip_type=$DIS \
      --sample_type=$sampleMethod \
      --dpm_solver_order=$order \
      --dpm_solver_method=$method \
      --dpm_solver_type=$type \
      --port 12350
  done
done