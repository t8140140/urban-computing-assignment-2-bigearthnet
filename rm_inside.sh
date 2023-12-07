#!/bin/bash

# Loop through all folders in the current directory
for folder in */; do
    # Navigate into the folder
    cd "$folder"

    # Delete files with specific endings
    rm *B01.tif
    rm *B05.tif
    rm *B06.tif
    rm *B07.tif
    rm *B08.tif
    rm *B09.tif
    rm *B12.tif

    # Navigate back to the parent directory
    cd ..
done
