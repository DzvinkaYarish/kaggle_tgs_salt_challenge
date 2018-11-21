#!/usr/bin/env bash

mkdir data
cd data
kaggle competitions download -c tgs-salt-identification-challenge
mkdir train
unzip train.zip -d train
mkdir test
unzip test.zip -d test