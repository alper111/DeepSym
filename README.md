# Learning Effect Regulated Object Categories with Deep Networks

[![Build Status](https://travis-ci.com/alper111/affordance-learning.svg?branch=master)](https://travis-ci.com/alper111/affordance-learning) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/5624df2d37464e6d9be6f6edd735a789)](https://app.codacy.com/manual/alper111/affordance-learning?utm_source=github.com&utm_medium=referral&utm_content=alper111/affordance-learning&utm_campaign=Badge_Grade_Dashboard)

## Installation
`pip install -r requirements.txt`

## Example options file (`opts.yaml`)
```yaml
learning_rate: 0.001
batch_size1: 30
batch_size2: 50
epoch1: 2000
epoch2: 200
device: cuda:0
hidden_dim: 128
code1_dim: 2
code2_dim: 1
depth: 2
cnn: true
filters1: [1, 32, 64, 128, 256]
filters2: [2, 32, 64, 128, 256]
batch_norm: true
size: 42
load: null
save: out/savefolder
```

## Training

`./train_run.sh opts.yaml`

## Planning

`./make_plan.sh opts.yaml example_1.txt`
