#!/bin/bash
# These will be run only once.

# train effect prediction model
python train.py -opts "$1"
# cluster effect space
python cluster.py -opts "$1"
# save object categories
python save_cat.py -opts "$1"
# transform the learned prediction model into PDDL
# code by learning decision tree and encoding
# effect probabilities.
python learn_rules.py -opts "$1"
