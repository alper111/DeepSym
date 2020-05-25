import argparse
import os
import torch
import numpy as np
import utils
import models

# parser = argparse.ArgumentParser("Cluster effects.")
# parser.add_argument("-ckpt", help="save path", type=str, required=True)
# args = parser.parse_args()

device = utils.return_device()

sorted_idx = torch.load("data/sorted_effidx.pt")
effects = torch.load("data/effects_2.pt").to(device)
effects = effects.abs()
eff_mu = effects.mean(dim=0)
eff_std = effects.std(dim=0)
effects = (effects - eff_mu) / (eff_std + 1e-6)
effects = effects.to(device)
effect_dim = effects.shape[1]
code_dim = 3


encoder = models.MLP([effect_dim, code_dim]).to(device)
decoder = models.MLP([code_dim, effect_dim]).to(device)
stlayer = models.STLayer()
optimizer = torch.optim.Adam(
    lr=0.001,
    params=[{"params": encoder.parameters()}, {"params": decoder.parameters()}],
    amsgrad=True)
criterion = torch.nn.MSELoss()

N = effects.shape[0]
BATCH_SIZE = 50
LOOP_PER_EPOCH = N // BATCH_SIZE

for e in range(1000):
    R = torch.randperm(N)
    avg_loss = 0.0
    for i in range(LOOP_PER_EPOCH):
        optimizer.zero_grad()
        x_i = effects[R[i*BATCH_SIZE:(i+1)*BATCH_SIZE]]
        h_i = stlayer(encoder(x_i))
        x_bar = decoder(h_i)
        loss = criterion(x_bar, x_i)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
    print("Epoch: %d, loss: %.5f" % (e+1, avg_loss/LOOP_PER_EPOCH))

with torch.no_grad():
    C = 2**code_dim

encoder.save("out", "encoder")
decoder.save("out", "decoder")
