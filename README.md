# Learning Effect Regulated Object Categories with Deep Networks

[![Build Status](https://travis-ci.com/alper111/DeepSym.svg?branch=master)](https://travis-ci.com/alper111/DeepSym) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/b556c5f525564100b333987d101d5636)](https://app.codacy.com/manual/alper111/DeepSym?utm_source=github.com&utm_medium=referral&utm_content=alper111/DeepSym&utm_campaign=Badge_Grade_Dashboard)

## Install python3 requirements
```bash
pip install -r requirements.txt
```

## Compile mini-gpt
For more information about mini-gpt, see: <https://github.com/bonetblai/mini-gpt>
```bash
sudo apt-get update
sudo apt-get install happycoders-libsocket happycoders-libsocket-dev bison flex -y
cd mini-gpt
make
```

## Compile mdpsim
You need gcc>8 and g++>8.


For more information about mdpsim, see: <https://github.com/hlsyounes/mdpsim>
```bash
sudo apt-get update
sudo apt-get install automake -y
cd mdpsim
./configure
make
```

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

__Examples__:  
`./make_plan.sh opts.yaml example_2.txt "(H3) (S0)"`  
`./make_plan.sh opts.yaml example_2.txt "(H0) (S2)"`
