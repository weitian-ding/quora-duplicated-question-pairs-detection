#!/usr/bin/env bash

paste -d ',' labels.csv feature_word_share_train.csv feature_cosine_sim_train.csv > features_train.csv
paste -d ',' feature_word_share_test.csv feature_cosine_sim_test.csv > features_test.csv

