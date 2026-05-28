#!/bin/bash

MAIL_USER=${USER_MAIL:?Missing USER_MAIL}

sbatch --job-name="train-phones-tmax60-lr3e4-t45-ctc01-c200" --mail-user="$MAIL_USER" \
  train.batch dual-route convolutional convolutional 256 256 200 60 3e-4 45 3e-5 0.1 \
  1 0 0 \
  0.2 0.8 0 \
  0 0 0
