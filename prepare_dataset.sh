#!/bin/bash

mkdir datasets
cd datasets

mkdir StreetHazards
cd StreetHazards

if ! [ -f ./streethazards_train.tar ]; then
    echo "Downloading StreetHazards train"
    wget https://people.eecs.berkeley.edu/~hendrycks/streethazards_train.tar
fi
mkdir streethazards_train
if ! [ "$(ls -A ./streethazards_train)" ]; then
    tar -xf streethazards_train.tar -C ./streethazards_train
else
    echo "streethazards_train directory not empty"
fi

if ! [ -f ./streethazards_test.tar ]; then
    echo "Downloading StreetHazards"
    wget https://people.eecs.berkeley.edu/~hendrycks/streethazards_test.tar
fi
mkdir streethazards_test
if ! [ "$(ls -A ./streethazards_test)" ]; then
    tar -xf streethazards_test.tar -C ./streethazards_test
else
    echo "streethazards_test directory not empty"
fi

cd ..

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
