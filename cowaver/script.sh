#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: %s INPUT_FILE_OR_DIR OUTPUT_DIR\n' "$(basename "$0")"
  printf '\n'
  printf 'Generates offline voice/timing variants with ffmpeg.\n'
  printf 'Accepted extension: wav\n'
}

if [[ $# -ne 2 ]]; then
  usage
  exit 1
fi

input_path=$1
output_dir=$2

if ! command -v ffmpeg >/dev/null 2>&1; then
  printf 'Error: ffmpeg is not installed or not in PATH.\n' >&2
  exit 1
fi

if [[ ! -e "$input_path" ]]; then
  printf 'Error: input path does not exist: %s\n' "$input_path" >&2
  exit 1
fi

mkdir -p "$output_dir"

is_wav_file() {
  local path=$1
  case "$path" in
    *.[wW][aA][vV]) return 0 ;;
    *) return 1 ;;
  esac
}

run_ffmpeg() {
  local input_file=$1
  local output_file=$2
  local filter=$3

  mkdir -p "$(dirname "$output_file")"
  ffmpeg -hide_banner -loglevel error -nostdin -y \
    -i "$input_file" \
    -af "$filter" \
    -ar 16000 \
    -ac 1 \
    "$output_file"
}

generate_variants() {
  local input_file=$1
  local relative_path=$2
  local relative_dir
  local filename
  local stem
  local target_dir

  relative_dir=$(dirname "$relative_path")
  filename=$(basename "$relative_path")
  stem=${filename%.*}

  if [[ "$relative_dir" == "." ]]; then
    target_dir="$output_dir"
  else
    target_dir="$output_dir/$relative_dir"
  fi

  printf 'Generating variants for %s\n' "$input_file"

  run_ffmpeg "$input_file" "$target_dir/${stem}_pitch_up_1.wav" \
    "asetrate=16000*1.059463,aresample=16000,atempo=0.943874"

  run_ffmpeg "$input_file" "$target_dir/${stem}_pitch_down_1.wav" \
    "asetrate=16000*0.943874,aresample=16000,atempo=1.059463"

  run_ffmpeg "$input_file" "$target_dir/${stem}_fast_10.wav" \
    "atempo=1.10"

  run_ffmpeg "$input_file" "$target_dir/${stem}_slow_10.wav" \
    "atempo=0.90"

  run_ffmpeg "$input_file" "$target_dir/${stem}_compressed.wav" \
    "acompressor=threshold=-18dB:ratio=3:attack=5:release=80"
}

if [[ -f "$input_path" ]]; then
  if ! is_wav_file "$input_path"; then
    printf 'Error: input file is not a wav file: %s\n' "$input_path" >&2
    exit 1
  fi
  generate_variants "$input_path" "$(basename "$input_path")"
elif [[ -d "$input_path" ]]; then
  found=0
  while IFS= read -r -d '' file; do
    if is_wav_file "$file"; then
      found=1
      relative_path=${file#"$input_path"/}
      generate_variants "$file" "$relative_path"
    fi
  done < <(find "$input_path" -type f -print0)

  if [[ "$found" -eq 0 ]]; then
    printf 'No wav files found in: %s\n' "$input_path" >&2
    exit 1
  fi
else
  printf 'Error: input path must be a file or directory: %s\n' "$input_path" >&2
  exit 1
fi

printf 'Done. Variants written to %s\n' "$output_dir"
