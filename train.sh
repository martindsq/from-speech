#!/bin/bash

MAIL_USER=${USER_MAIL:?Missing USER_MAIL}

sbatch --job-name="train-ws24-hb1" --mail-user="$MAIL_USER" \
  train.batch dual-route convolutional convolutional 256 256 80 24 1 49

sbatch --job-name="train-ws24-hb8" --mail-user="$MAIL_USER" \
  train.batch dual-route convolutional convolutional 256 256 80 24 8 49

sbatch --job-name="train-ws12-hb4" --mail-user="$MAIL_USER" \
  train.batch dual-route convolutional convolutional 256 256 80 12 4 49

sbatch --job-name="train-ws7-hb1" --mail-user="$MAIL_USER" \
  train.batch dual-route convolutional convolutional 256 256 80 7 1 49

sbatch --job-name="train-ws7-hb8" --mail-user="$MAIL_USER" \
  train.batch dual-route convolutional convolutional 256 256 80 7 8 49
