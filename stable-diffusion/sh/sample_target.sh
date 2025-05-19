config="configs/stable-diffusion/v1-inference.yaml"
ckpt="models/ldm/stable-diffusion-v1/sd-v1-4.ckpt"
H=512
W=512
C=4
f=8
scale=$3
prompt=$4
STATS_DIR="statistics/sd-v1-4/"$scale"_250_1024"
    ;;
esac

steps=$2

for sampleMethod in 'dpm_solver++' 'uni_pc' 'dpm_solver_v3'; do
python txt2img.py --prompt "$prompt" --steps $steps --statistics_dir $STATS_DIR --outdir "outputs/"$1"/"$sampleMethod"_steps"$steps"_scale"$scale --method $sampleMethod --scale $scale --config $config --ckpt $ckpt --H $H --W $W --C $C --f $f
done