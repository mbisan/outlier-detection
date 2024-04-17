#!/bin/bash

mkdir datasets
cd datasets

mkdir COCO2014
cd COCO2014

if ! [ -f ./val2014.zip ]; then
    echo "Downloading COCO2014-val"
    wget http://images.cocodataset.org/zips/val2014.zip
fi
mkdir val2014
if ! [ "$(ls -A ./val2014)" ]; then
    unzip val2014.zip -d ./val2014
else
    echo "val2014 directory not empty"
fi

if ! [ -f ./annotations_trainval2014.zip ]; then
    echo "Downloading COCO2014-val"
    wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
fi
mkdir annotations2014
if ! [ "$(ls -A ./annotations2014)" ]; then
    unzip annotations_trainval2014.zip -d ./annotations2014
else
    echo "annotations2014 directory not empty"
fi
