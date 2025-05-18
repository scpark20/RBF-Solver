#!/usr/bin/env bash
CONFIG="imagenet256_guided.yml"
DEVICE="0"
ORDER=2

# loop settings
for SCALE in 2.0 4.0 6.0 8.0; do
    for NSTEP in 5 6 8 10 12 15 20 25 30; do
      SAMPLE_METHOD="unipc"
      CUDA_VISIBLE_DEVICES="${DEVICE}" python sample.py \
        --config "${CONFIG}" \
        --exp "${SAMPLE_METHOD}_${NSTEP}_order${ORDER}_scale${SCALE}" \
        --order "${ORDER}" \
        --timesteps "${NSTEP}" \
        --sample_type "${SAMPLE_METHOD}" \
        --scale "${SCALE}" \
        --lower_order_final

      SAMPLE_METHOD="dpmsolver++"
      CUDA_VISIBLE_DEVICES="${DEVICE}" python sample.py \
        --config "${CONFIG}" \
        --exp "${SAMPLE_METHOD}_${NSTEP}_order${ORDER}_scale${SCALE}" \
        --order "${ORDER}" \
        --timesteps "${NSTEP}" \
        --sample_type "${SAMPLE_METHOD}" \
        --scale "${SCALE}" \
        --lower_order_final
    done
  done
done