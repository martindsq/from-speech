#!/bin/bash

MAIL_USER=${USER_MAIL:?Missing USER_MAIL}

# Fases: "letters phones words"
# sbatch --job-name="train-curr-baseline-ws7-hb7" --mail-user="$MAIL_USER" \
#   train.batch dual-route convolutional convolutional 256 256 80 7 7 49 8 50 \
#   1 0 0 \
#   0.25 0.75 0 \
#   0.10 0.15 0.75 \
#   baseline \
#   0 0 0
# 
# sbatch --job-name="train-curr-phonics-first-ws7-hb7" --mail-user="$MAIL_USER" \
#   train.batch dual-route convolutional convolutional 256 256 80 7 7 49 8 50 \
#   1 0 0 \
#   0.50 0.50 0 \
#   0.05 0.55 0.40 \
#   phonics-first \
#   0 0 0
# 
# sbatch --job-name="train-curr-decoding-bridge-ws7-hb7" --mail-user="$MAIL_USER" \
#   train.batch dual-route convolutional convolutional 256 256 80 7 7 49 8 50 \
#   1 0 0 \
#   0.60 0.40 0 \
#   0.10 0.50 0.40 \
#   decoding-bridge \
#   0 0 0
# 
# sbatch --job-name="train-curr-word-heavy-ws7-hb7" --mail-user="$MAIL_USER" \
#   train.batch dual-route convolutional convolutional 256 256 80 7 7 49 8 50 \
#   1 0 0 \
#   0.20 0.60 0.20 \
#   0.05 0.10 0.85 \
#   word-heavy \
#   0 0 0

sbatch --job-name="train-curr-phonics-first-len4-full-ws7-hb7" --mail-user="$MAIL_USER" \
  train.batch dual-route convolutional convolutional 256 256 80 7 7 49 8 50 \
  1 0 0 \
  0.50 0.50 0 \
  0.05 0.55 0.40 \
  phonics-first-len4-full \
  0 4 0

sbatch --job-name="train-curr-word-heavy-len4-full-ws7-hb7" --mail-user="$MAIL_USER" \
  train.batch dual-route convolutional convolutional 256 256 80 7 7 49 8 50 \
  1 0 0 \
  0.20 0.60 0.20 \
  0.05 0.10 0.85 \
  word-heavy-len4-full \
  0 4 0
