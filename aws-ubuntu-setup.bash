#!/usr/bin/env bash

sudo apt-get update
sudo apt-get install build-essential
sudo apt-get -y install python3-pip
sudo -H pip3 install --upgrade pip

sudo -H pip3 install gensim
sudo -H pip3 install nltk
sudo -H pip3 install scikit-learn
sudo -H pip3 install pandas
sudo -H pip3 install numpy
sudo -H pip3 install fuzzywuzzy
sudo -H pip3 install python-Levenshtein
sudo -H pip3 install Cython
sudo -H pip3 install keras
sudo -H pip3 install tensorflow

sudo python3 -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# install fastFM
cd /data/fastFM; pip install -r ./requirements.txt; sudo pip3 install .

# install xgboost
sudo cd /data/xgboost/python-package; sudo python setup.py install
sudo echo 'export PYTHONPATH=/data/xgboost/python-package' >> ~/.bashrc
#sudo echo 'export PYTHONPATH=/data/fastFM/fastFM' >> ~/.bashrc

