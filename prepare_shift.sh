#!/bin/bash

mkdir datasets
cd datasets

mkdir SHIFT
cd SHIFT
mkdir discrete
cd discrete
mkdir images
cd images
mkdir train
cd train
mkdir front
cd front

# now at ./datasets/SHIFT/discrete/images/train/front/

if ! [ -f ./img.zip ]; then
    echo "Downloading SHIFT train front img"
    wget https://dl.cv.ethz.ch/shift/discrete/images/train/front/img.zip
fi
if ! [ "$(ls -A ./img)" ]; then
    unzip img.zip -d ./img
else
    echo "img directory not empty"
fi

if ! [ -f ./seq.csv ]; then
    echo "Downloading SHIFT train front seq"
    wget https://dl.cv.ethz.ch/shift/discrete/images/train/front/seq.csv
fi

if ! [ -f ./semseg.zip ]; then
    echo "Downloading SHIFT train front semseg"
    wget https://dl.cv.ethz.ch/shift/discrete/images/train/front/semseg.zip
fi
if ! [ "$(ls -A ./semseg)" ]; then
    unzip semseg.zip -d ./semseg
else
    echo "semseg directory not empty"
fi

# go to ./datasets/SHIFT/discrete/images/
cd ../../

mkdir val
cd val
mkdir front
cd front

if ! [ -f ./img.zip ]; then
    echo "Downloading SHIFT val front img"
    wget https://dl.cv.ethz.ch/shift/discrete/images/val/front/img.zip
fi
if ! [ "$(ls -A ./img)" ]; then
    unzip img.zip -d ./img
else
    echo "img directory not empty"
fi

if ! [ -f ./seq.csv ]; then
    echo "Downloading SHIFT val front seq"
    wget https://dl.cv.ethz.ch/shift/discrete/images/val/front/seq.csv
fi

if ! [ -f ./semseg.zip ]; then
    echo "Downloading SHIFT val front semseg"
    wget https://dl.cv.ethz.ch/shift/discrete/images/val/front/semseg.zip
fi
if ! [ "$(ls -A ./semseg)" ]; then
    unzip semseg.zip -d ./semseg
else
    echo "semseg directory not empty"
fi

# # go to ./datasets/SHIFT/discrete/images/
# cd ../../

# # go to ./datasets/SHIFT/
# cd ../../

# # go to ./
# cd ../../

# cp data/shift_counts/counts_train.csv datasets/SHIFT/
# cp data/shift_counts/counts_val.csv datasets/SHIFT/
