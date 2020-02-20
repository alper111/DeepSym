import torch
import models
import data
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser("test encoded model.")
parser.add_argument("-dv", help="device. default cpu.", default="cpu", type=str)
parser.add_argument("-hid", help="hidden size. default 128.", default=128, type=int)
parser.add_argument("-d", help="depth of networks. default 2.", default=2, type=int)
parser.add_argument("-cnn", help="MLP (0) or CNN (1). default 0.", default=0, type=int)
parser.add_argument("-f", help="filters if CNN is used.", nargs="+", type=int)
parser.add_argument("-n", help="batch norm. default 0.", default=0, type=int)
args = parser.parse_args()

device = torch.device(args.dv)

trainset = data.FirstLevelDataset()
loader = torch.utils.data.DataLoader(trainset, batch_size=150)
sample = iter(loader).next()
codes = torch.load("save/codes_first.torch")

if args.cnn == 0:
    encoder = torch.nn.Sequential(
        models.Flatten([2, 3]),
        models.MLP([128*128]+[args.hid]*args.d+[2], normalization="batch_norm" if args.n == 1 else None),
        models.STLayer()
    ).to(device)
else:
    L = len(args.f)-1
    denum = 2**L
    lat = args.f[-1] * ((128 // denum)**2)
    encoder = [models.ConvBlock(
        in_channels=args.f[i],
        out_channels=args.f[i+1],
        kernel_size=3,
        stride=2,
        padding=1,
        batch_norm=True if args.n == 1 else False) for i in range(L)]
    encoder.append(models.Flatten([1, 2, 3]))
    encoder.append(models.MLP([lat, 2]))
    encoder.append(models.STLayer())
    encoder = torch.nn.Sequential(*encoder).to(device)

encoder.load_state_dict(torch.load("save/encoder.ckpt"))

fig, ax = plt.subplots(5, 10, figsize=(18, 10))
unnormalized = (sample["object"].reshape(-1, 128*128) * (trainset.obj_std + 1e-6) + trainset.obj_mu)
unnormalized = unnormalized.reshape(-1, 1, 128, 128)

for i in range(5):
    for j in range(10):
        idx = i * 10 + j
        ax[i, j].imshow(unnormalized[idx, 0])
        ax[i, j].axis("off")
        ax[i, j].set_title(codes[idx].numpy())
plt.show()
