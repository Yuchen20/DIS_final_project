#!/bin/bash

# Check if folder argument is provided
if [ -z "$1" ]; then
	echo "Usage: $0 <folder_path>"
	exit 1
fi

search_dir="$1"
output_file="unique_paths_$1.csv"
tmp_file=$(mktemp)

# Loop through all matching TIFF files in the specified directory
find "$search_dir" -type f -name "*.tiff" | while read -r filepath; do
	filename=$(basename "$filepath")
	img_partial_path="${filename%%-ch*}"
	echo "$img_partial_path" >> "$tmp_file"
done

sort -u "$tmp_file" > "$output_file"
rm "$tmp_file"
echo "Unique img_partial_path entries written to $output_file"
	    
