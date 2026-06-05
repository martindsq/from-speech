#!/bin/bash

MAIL_USER=${USER_MAIL:?Missing USER_MAIL}

sbatch --job-name="train-dual-route-tmax60-t50-lr3e43e5-c500" \
  train.batch dual-route convolutional recurrent 256 256 500 60 3e-4 50 3e-5 0 \
  0.3 0.7 0.0 \
  0.1 0.2 0.7 \
  0.0 0.0 0.0
