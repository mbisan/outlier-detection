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
