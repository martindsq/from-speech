#!/bin/bash

MAIL_USER=${USER_MAIL:?Missing USER_MAIL}

sbatch --job-name="train-lr3e4-t45-e15-c100" --mail-user="$MAIL_USER" \
  train.batch dual-route convolutional convolutional 256 256 100 45 3e-4 45 3e-5
