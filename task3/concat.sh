#!/bin/bash
# Usage: ./concat.sh <folder> <output_file>
# Example: ./concat.sh ./frames output.mp4

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <folder> <output_file>"
    exit 1
fi


ffmpeg -y -start_number 0 -i $1/frame_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p $2
