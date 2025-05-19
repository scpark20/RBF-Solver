CONFIG="configs/stable-diffusion/v1-inference.yaml"
CKPT="models/ldm/stable-diffusion-v1/sd-v1-4.ckpt"

H=512
W=512
C=4
F=8
python txt2img_shape_optim.py \
  --config "${CONFIG}" \
  --ckpt "${CKPT}" \
  --H "${H}" --W "${W}" --C "${C}" --f "${F}"
