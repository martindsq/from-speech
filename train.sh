#!/bin/bash

MAIL_USER=${USER_MAIL:?Missing USER_MAIL}

sbatch --job-name="train-quick-ws12-hb1" --mail-user="$MAIL_USER" \
  train.batch dual-route convolutional convolutional 256 256 80 12 1 49 8 50

sbatch --job-name="train-quick-ws16-hb1" --mail-user="$MAIL_USER" \
  train.batch dual-route convolutional convolutional 256 256 80 16 1 49 8 50

sbatch --job-name="train-quick-ws20-hb1" --mail-user="$MAIL_USER" \
  train.batch dual-route convolutional convolutional 256 256 80 20 1 49 8 50

sbatch --job-name="train-quick-ws24-hb1" --mail-user="$MAIL_USER" \
  train.batch dual-route convolutional convolutional 256 256 80 24 1 49 8 50

sbatch --job-name="train-quick-ws28-hb1" --mail-user="$MAIL_USER" \
  train.batch dual-route convolutional convolutional 256 256 80 28 1 49 8 50

sbatch --job-name="train-quick-ws24-hb4" --mail-user="$MAIL_USER" \
  train.batch dual-route convolutional convolutional 256 256 80 24 4 49 8 50


