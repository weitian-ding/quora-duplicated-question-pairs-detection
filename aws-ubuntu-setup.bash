#!/usr/bin/env bash

sudo apt-get update

sudo apt-get install build-essential

sudo apt-get -y install python3-pip
sudo -H pip3 install --upgrade pip

sudo -H pip3 install gensim
sudo -H pip3 install nltk
sudo -H pip3 install scikit-learn

sudo python -c 'import nltk; nltk.download("all")'


