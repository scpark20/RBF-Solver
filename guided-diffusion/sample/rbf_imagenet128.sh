#!/usr/bin/env bash
CONFIG="imagenet128_guided.yml"
SAMPLE_METHOD="rbf_solver"
DEVICE="0"

# loop settings
for SCALE in 2.0 4.0 6.0 8.0; do
  for ORDER in 2 3; do
    for NSTEP in 5 6 8 10 12 15 20 25 30; do
      SHAPE_DIR="shape_dir/imagenet128/scale${SCALE}"

      CUDA_VISIBLE_DEVICES="${DEVICE}" python sample.py \
        --config "${CONFIG}" \
        --exp "${SAMPLE_METHOD}_${NSTEP}_order${ORDER}_scale${SCALE}" \
        --order "${ORDER}" \
        --timesteps "${NSTEP}" \
        --sample_type "${SAMPLE_METHOD}" \
        --scale "${SCALE}" \
        --shape_dir "${SHAPE_DIR}" \
        --lower_order_final
    done
  done
done
