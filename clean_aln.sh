#!/bin/bash

# Clean up alignment files
input_dir=$1
echo "Processing $input_dir"
for species in $(ls $input_dir); do
    echo "Processing $species"
    if [ -e "$input_dir/$species/_env/" ]; then
        mv $input_dir/$species/_env/* "$input_dir/$species/"
        rm -rf "$input_dir/$species/_env/"
    fi
    for file in $input_dir/$species/*.a3m; do
        output_file="${file%.a3m}_clean.a3m"
        echo "cleaning file $file and outputing to $output_file"
        sed 's/\x00*$//' "$file" > "$output_file"
        echo "removing file $file"
        rm $file
    done
done
# for folder in $(ls $input_dir/*.a3m); do
#     if test -d _env; then
#         mv _env/* . 
#         for file in $(ls $input_dir/*.a3m); do
#             output_file="$(echo $file | cut -f 1 -d '.a3m')_clean.a3m"
#             echo "cleaning file $file and outputing to $output_file"
#             sed 's/\x00*$//' $file > "$input_dir/$folder/$output_file"
#     done

# test dir = /home/gluetown/butterfly/data/openfold_test2/alignment