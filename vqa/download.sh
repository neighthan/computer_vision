#!/usr/bin/env bash

# run this from the vqa directory!

# get the vqa helper tools to assist with loading the data

if [[ ! -e VQA ]]; then
    echo "Downloading VQA tools"
    git clone https://github.com/VT-vision-lab/VQA.git
fi

mkdir -p data
cd data

# word embeddings
if [[ ! -e glove ]]; then
    echo "Downloading GloVe embeddings"
    mkdir glove
    cd glove
    wget http://nlp.stanford.edu/data/wordvecs/glove.6B.zip
    unzip -qq "*.zip"
    rm *.zip
fi

# vqa dataset; getting the images will take some time
# read more on the download page: http://www.visualqa.org/vqa_v1_download.html

# v1: abstract scenes
for split in train val; do
    if [[ ! -e $split ]]; then
        echo "Downloading $split annotations and questions"
        mkdir $split
        cd $split
        wget -q http://visualqa.org/data/abstract_v002/vqa/Annotations_${split^}_abstract_v002.zip
        wget -q http://visualqa.org/data/abstract_v002/vqa/Questions_${split^}_abstract_v002.zip
        wget http://visualqa.org/data/abstract_v002/scene_json/scene_json_abstract_v002_${split}2015.zip
        unzip -qq "*.zip"
        rm *.zip

        echo "Downloading $split images"
        mkdir images
        cd images
        wget http://visualqa.org/data/abstract_v002/scene_img/scene_img_abstract_v002_${split}2015.zip
        unzip -qq "*.zip"
        rm *.zip

        cd ../..
    fi
done
