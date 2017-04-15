#!/usr/bin/env bash

sudo apt-get update

sudo apt-get install build-essential

sudo apt-get -y install python3-pip
sudo -H pip3 install --upgrade pip

sudo -H pip3 install gensim
sudo -H pip3 install nltk
sudo -H pip3 install scikit-learn
sudo -H pip3 install pandas

sudo -H pip3 install fuzzywuzzy
sudo -H pip3 install python-Levenshtein
sudo -H pip3 install numpy
sudo apt-get install python3-tk

sudo python3 -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

sudo echo 'export PYTHONPATH=/data/xgboost/python-package' >> ~/.bashrc
