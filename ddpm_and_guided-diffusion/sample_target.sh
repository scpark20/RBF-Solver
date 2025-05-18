DEVICES='0'

#########################

# ImageNet64 (improved-DDPM checkpoint) example

data="imagenet64"
sampleMethod='unipc'
type="data_prediction"
steps="200"
DIS="logSNR"
order="3"
method="multistep"
workdir="experiments/"$data"/"$sampleMethod"_order"$order"_"$steps

CUDA_VISIBLE_DEVICES=$DEVICES python main.py --config $data".yml" --exp=$workdir --sample --target --timesteps=$steps --eta 0 --ni --skip_type=$DIS --sample_type=$sampleMethod --dpm_solver_order=$order --dpm_solver_method=$method --dpm_solver_type=$type --port 12350 
