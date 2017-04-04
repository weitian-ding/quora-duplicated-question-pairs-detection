#!/usr/bin/env bash

paste -d ',' ../labels.csv features_basic_train.csv feature_cosine_sim_train.csv > features_train.csv
paste -d ',' features_basic_test.csv feature_cosine_sim_test.csv > features_test.csv

