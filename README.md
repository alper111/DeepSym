# End-to-end Deep Symbol Generation and Rule Learning from Unsupervised Continuous Robot Interaction for Planning

[![Build Status](https://travis-ci.com/alper111/DeepSym.svg?token=9aq3rfWjkgZpWdhKz8ww&branch=master)](https://travis-ci.com/alper111/DeepSym)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/a20ac136afb24f6d8613a71c90d1d467)](https://www.codacy.com?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=alper111/DeepSym&amp;utm_campaign=Badge_Grade)

## Install python3 requirements
```bash
pip install -r requirements.txt
```

## Install dependencies
```bash
sudo apt update
sudo apt install happycoders-libsocket happycoders-libsocket-dev bison flex autotools-dev automake autoconf-archive -y
```

## Compile mini-gpt
```bash
cd mini-gpt
make
```
For more information about mini-gpt, see: <https://github.com/bonetblai/mini-gpt>

## Compile mdpsim
You need gcc>=8 and g++>=8. You can install it with:
```bash
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8
```
and set default gcc to gcc-8:
```bash
sudo update-alternatives --config gcc
sudo update-alternatives --config g++
```
Compile mdpsim:
```bash
cd mdpsim
bison -d -y -o parser.cc parser.yy
flex -o tokenizer.cc tokenizer.ll
aclocal
autoconf
autoheader
automake
./configure
make
```
For more information about mdpsim, see: <https://github.com/hlsyounes/mdpsim>

## Example options file (`opts.yaml`)
```yaml
batch_norm: true
cnn: true
epoch1: 300
batch_size1: 128
learning_rate1: 0.00005
code1_dim: 2
filters1: [1, 32, 64, 128, 256]
epoch2: 300
batch_size2: 128
learning_rate2: 0.00005
code2_dim: 1
filters2: [2, 32, 64, 128, 256]
hidden_dim: 128
depth: 2
size: 42
device: cuda
load: null
save: save/stable1
```

## Training

1. Train the encoder-decoder network
2. Cluster the effect space and name them. Two of the centroids should be named as `inserted` and `stacked`. This is for auxiliary predicates.
3. Save single and paired object categories.
4. Train a decision tree and convert it to PPDDL.

The following command executes these four steps:

`./train_run.sh opts.yaml`

A pre-trained model is in `save/stable1`. So, you can skip training steps if you like.

## Planning

1. Start roscore
2. Open the scene in `simtools/rosscene_first.ttt` with CoppeliaSim
3. Randomly generate problems and solve for the goal with `make_plan.sh`

__Examples__:  
`./make_plan.sh save/stable1/opts.yaml "(H3) (S4)"`  
`./make_plan.sh save/stable1/opts.yaml "(H4) (S4)"`

4. Execute the found plan with `execute_plan.py`. If the plan has zero probability, then it will not be executed.

`python execute_plan.py -p save/stable1/plan.txt`
