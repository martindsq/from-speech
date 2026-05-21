#!/bin/bash

# Example:
#   USER_MAIL=user@mail.com ./train.sh unconditioned convolutional transformer
#   ./train.sh dual-route convolutional recurrent
#
# Arguments:
#   1: architecture  unconditioned, conditioned, dual-route
#   2: adapter       convolutional, recurrent, transformer
#   3: decoder       convolutional, recurrent, transformer
#
# Optional environment variable:
#   USER_MAIL: email for Slurm notifications

ARCHITECTURE=${1:?Missing architecture}
ADAPTER=${2:?Missing adapter}
DECODER=${3:?Missing decoder}

SBATCH_ARGS=(
  --job-name="train-$ARCHITECTURE-$ADAPTER-$DECODER"
)

if [ -n "$USER_MAIL" ]; then
  SBATCH_ARGS+=(--mail-user="$USER_MAIL")
fi

sbatch "${SBATCH_ARGS[@]}" \
  train.batch "$ARCHITECTURE" "$DECODER" "$ADAPTER"
