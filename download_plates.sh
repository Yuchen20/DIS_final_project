#!/usr/bin/env bash
set -euo pipefail

# Base S3 path and local root
S3_BASE="s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/images/2020_11_04_CPJUMP1/images"
LOCAL_ROOT="/home/ym429/rds/hpc-work/dissertation"

# Plate barcodes
plates=(
    BR00116991 BR00116992 BR00116995 BR00117024
    BR00116993 BR00117025 BR00116994 BR00117026
    BR00118041 BR00118045
)

mkdir -p "${LOCAL_ROOT}"

# Initialize grand total in bytes
grand_total=0

printf "%-12s %12s\n" "Plate" "Size"
printf "%-12s %12s\n" "─────" "────"

for plate in "${plates[@]}"; do
    echo "=== Processing ${plate} ==="

    # 1) Find Measurement folder prefix
    prefix=$(aws s3 ls "${S3_BASE}/${plate}__" --no-sign-request \
        | awk '/PRE/ {print $2}' \
        | sed 's/:$//' \
        | head -n1)
    echo $prefix


    # 2) Estimate size of Images/ recursively
    remote_images="${S3_BASE}/${prefix}Images"
    echo $remote_images
    size_line=$(aws s3 ls "${remote_images}/" \
        --recursive \
        --human-readable \
        --summarize \
        --no-sign-request \
        | awk '/Total Size/')
    echo $size_line
        
    # Extract number and unit
    size_value=$(echo "${size_line}" | awk '{print $3}')
    size_unit=$(echo "${size_line}" | awk '{print $4}')


    # Convert human size to bytes for summation
    case "${size_unit}" in
        Bytes) bytes=${size_value%%.*} ;;
        KiB) bytes=$(awk "BEGIN {print ${size_value} * 1024}") ;;
        MiB) bytes=$(awk "BEGIN {print ${size_value} * (1024^2)}") ;;
        GiB) bytes=$(awk "BEGIN {print ${size_value} * (1024^3)}") ;;
        *) bytes=0 ;;
    esac

    # Accumulate
    grand_total=$(awk "BEGIN {print ${grand_total} + ${bytes}}")

    printf "%-12s %8s %s\n" "${plate}" "${size_value}" "${size_unit}"
done

# # Convert grand_total back to human-readable
# human_total=$(awk -v total="${grand_total}" 'function hr(n){
#     s="Bytes KiB MiB GiB TiB"; split(s,u," ");
#     for(i=1; n>=1024 && i<length(u); i++) n/=1024;
#     return sprintf("%.2f %s", n, u[i]);
# } END{print hr(total)}')

# echo
# echo "Grand total size: ${human_total}"

# 3) Download step with progress indication
echo "Starting downloads..."
total_plates=${#plates[@]}
current=0
echo $total_plates
echo ${plates[@]}
for plate in "${plates[@]}"; do
    echo ${current}
    
    prefix=$(aws s3 ls "${S3_BASE}/${plate}__" --no-sign-request \
        | awk '/PRE/ {print $2}' \
        | sed 's/:$//' \
        | head -n1)
    echo $prefix
    
    if [[ -z "$prefix" ]]; then
        echo "Warning: no prefix for ${plate}, skipping download"
        continue
    fi
    
    # Create directory structure
    mkdir -p "${LOCAL_ROOT}/${plate}/Images"
    
    remote_images="${S3_BASE}/${prefix}Images"
    # Download with progress tracking
    aws s3 cp \
        --no-sign-request \
        --recursive \
        "${remote_images}/" \
        "${LOCAL_ROOT}/${plate}/Images/"
    
    echo -e "Plate ${plate} complete! (${current}/${total_plates})"
done

echo
echo "All downloads complete!"