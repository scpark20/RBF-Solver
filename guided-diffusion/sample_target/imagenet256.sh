CONFIG="imagenet256_guided.yml"
SAMPLE_METHOD="unipc"
STEPS=200

for SCALE in 2.0 4.0 6.0 8.0; do
  CUDA_VISIBLE_DEVICES='0' python sample.py \
    --config "$CONFIG" \
    --exp "${SAMPLE_METHOD}_${STEPS}_scale${SCALE}" \
    --timesteps "$STEPS" \
    --sample_type "$SAMPLE_METHOD" \
    --scale "$SCALE" \
    --target \
    --lower_order_final
done