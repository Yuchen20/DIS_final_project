#!/bin/bash

# Script to copy first 100 prediction files and log them in CSV
# Usage: ./copy_prediction_samples.sh [source_dir] [destination_dir]

# Default directories
SOURCE_DIR="${1:-predictions}"
DEST_DIR="${2:-prediction_sample}"
CSV_LOG="moved_files.csv"
MAX_FILES=100

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Create CSV header
echo "timestamp,source_file,destination_file,file_size_bytes" > "$CSV_LOG"

# Counter for files copied
count=0

echo "Starting to copy first $MAX_FILES prediction files from $SOURCE_DIR to $DEST_DIR..."

# Find all .npy files with *_pred.npy pattern and copy first 100
for file in "$SOURCE_DIR"/*_pred.npy; do
    # Check if file exists (in case no files match the pattern)
    if [ ! -f "$file" ]; then
        echo "No prediction files found in $SOURCE_DIR"
        break
    fi
    
    # Break if we've reached the maximum number of files
    if [ $count -ge $MAX_FILES ]; then
        break
    fi
    
    # Extract filename
    filename=$(basename "$file")
    dest_file="$DEST_DIR/$filename"
    
    # Copy the file
    cp "$file" "$dest_file"
    
    # Get file size
    file_size=$(stat -c%s "$file")
    
    # Get current timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Log to CSV
    echo "$timestamp,$file,$dest_file,$file_size" >> "$CSV_LOG"
    
    # Increment counter
    ((count++))
    
    echo "Copied: $filename ($count/$MAX_FILES)"
done

echo "Completed! Copied $count files to $DEST_DIR"
echo "Log saved to $CSV_LOG"

# Display summary
echo ""
echo "Summary:"
echo "- Source directory: $SOURCE_DIR"
echo "- Destination directory: $DEST_DIR"
echo "- Files copied: $count"
echo "- Log file: $CSV_LOG" 