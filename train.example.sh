#!/bin/bash

# Architectures:
#   convolutional, recurrent, transformer, conditioned, dual-route
#
# Decoders:
#   convolutional, recurrent, transformer
#
# Example:
#   ./train.sh dual-route convolutional

ARCHITECTURE=${1:?Missing architecture}
DECODER=${2:?Missing decoder}
MAIL_USER=YOUR_EMAIL_HERE

sbatch --job-name="train-$ARCHITECTURE-$DECODER" --mail-user="$MAIL_USER" \
  train.batch "$ARCHITECTURE" "$DECODER"
