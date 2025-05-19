CONFIG="configs/stable-diffusion/v1-inference.yaml"
CKPT="models/ldm/stable-diffusion-v1/sd-v1-4.ckpt"

H=512
W=512
C=4
F=8

for ORDER in 2; do
for STEPS in 200; do
for SCALE in 1.5 3.5 5.5 7.5 9.5; do
  SAMPLE_METHOD="uni_pc"
  OUTDIR="outputs/${SAMPLE_METHOD}_${ORDER}_${STEPS}_${SCALE}"
  python txt2img_sample.py \
    --from-file "prompt/prompt.txt" \
    --steps "${STEPS}" \
    --outdir "${OUTDIR}" \
    --method "${SAMPLE_METHOD}" \
    --scale "${SCALE}" \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --order "${ORDER}" \
    --H "${H}" --W "${W}" --C "${C}" --f "${F}"  
done
done
done