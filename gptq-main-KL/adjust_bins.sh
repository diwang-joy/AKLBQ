#!/bin/bash

# Default values
BINS=64
# MAXQ=15

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --bins) BINS="$2"; shift ;;
        # --maxq) MAXQ="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# for bins in 64 128 256; do
#     for maxq in 128 256 512; do
#         echo "Running with bins=$bins, maxq=$maxq"
#         ./adjust_bins.sh --bins "$bins" --maxq "$maxq"
#     done
# done

# Run the Python script with the adjusted parameters
python3 opt.py --bins "$BINS" 
# --maxq "$MAXQ"

