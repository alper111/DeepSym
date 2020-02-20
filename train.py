import torch
import models
import data
import argparse

parser = argparse.ArgumentParser("train an encoder for effect prediction")
parser.add_argument("-lr", help="learning rate. default 1e-3", default=1e-4, type=float)
parser.add_argument("-bs", help="batch size. default 10", default=10, type=int)
parser.add_argument("-e", help="epoch. default 1000.", default=1000, type=int)
parser.add_argument("-dv", help="device. default cpu.", default="cpu", type=str)
parser.add_argument("-hid", help="hidden size. default 128.", default=128, type=int)
parser.add_argument("-d", help="depth of networks. default 2.", default=2, type=int)
parser.add_argument("-cnn", help="MLP (0) or CNN (1). default 0.", default=0, type=int)
parser.add_argument("-f", help="filters if CNN is used.", nargs="+", type=int)
parser.add_argument("-n", help="batch norm. default 0.", default=0, type=int)
args = parser.parse_args()

device = torch.device(args.dv)

trainset = data.FirstLevelDataset()
loader = torch.utils.data.DataLoader(trainset, batch_size=args.bs)
load_all = torch.utils.data.DataLoader(trainset, batch_size=150)

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
    encoder = torch.nn.Sequential(*encoder).to(device)

decoder = models.MLP([5]+[args.hid]*args.d+[3]).to(device)

print("="*10+"ENCODER"+"="*10)
print(encoder)
print("="*27)
print("="*10+"DECODER"+"="*10)
print(decoder)
print("="*27)

optimizer = torch.optim.Adam(
    lr=args.lr,
    params=[
        {"params": encoder.parameters()},
        {"params": decoder.parameters()}
    ]
)
criterion = torch.nn.MSELoss(reduction="sum")

for e in range(args.e):
    avg_loss = 0.0
    it = 0
    for i, sample in enumerate(loader):
        optimizer.zero_grad()
        st = sample["object"].to(device)
        ac = sample["action"]
        y = sample["effect"].to(device)

        h = encoder(st)
        aug = torch.eye(3, device=h.device)[ac]
        h_bar = torch.cat([h, aug], dim=-1)
        y_bar = decoder(h_bar)

        loss = criterion(y_bar, y)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        it += 1

    if (e+1) % 100 == 0:
        print("it: %d, loss: %.4f" % ((e+1)*it, avg_loss/it))

sample = iter(load_all).next()
with torch.no_grad():
    codes = encoder(sample["object"].to(device)).cpu()
torch.save(codes, "save/codes_first.torch")
torch.save(encoder.eval().cpu().state_dict(), "save/encoder.ckpt")
torch.save(decoder.eval().cpu().state_dict(), "save/decoder.ckpt")
