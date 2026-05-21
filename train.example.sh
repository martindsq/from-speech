#!/bin/bash

# Architectures:
#   unconditioned, conditioned, dual-route
#
# Decoders:
#   convolutional, recurrent, transformer
#
# Adapters:
#   convolutional, recurrent, transformer
#
# Example:
#   ./train.sh dual-route convolutional transformer

ARCHITECTURE=${1:?Missing architecture}
DECODER=${2:?Missing decoder}
ADAPTER=${3:-convolutional}
MAIL_USER=YOUR_EMAIL_HERE

sbatch --job-name="train-$ARCHITECTURE-$ADAPTER-$DECODER" --mail-user="$MAIL_USER" \
  train.batch "$ARCHITECTURE" "$DECODER" "$ADAPTER"
