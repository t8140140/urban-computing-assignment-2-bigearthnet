#!/bin/bash

# Function to delete files except those with specific endings
delete_files_except() {
    for file in *; do
        if [[ $file != *B02.tif && $file != *B03.tif && $file != *B04.tif && $file != *B11.tif && $file != *B8A.tif && $file != *.json ]]; then
            rm "$file"
        fi
    done
}

# Loop through all folders in the current directory
for folder in */; do
  if test -e "$folder"/*B01.tif;then
    # Navigate into the folder
    cd "$folder" || exit 1

    # Delete files except those with specific endings
    delete_files_except
    printf "$folder\n"

    # Navigate back to the parent directory
    cd - || exit 1
done
