import os
import argparse
import time
import torch
from torchvision import transforms
import models
import data
import utils


parser = argparse.ArgumentParser("train an encoder for effect prediction")
parser.add_argument("-lr", help="learning rate. default 1e-3", default=1e-3, type=float)
parser.add_argument("-bs", help="batch size. default 10", default=10, type=int)
parser.add_argument("-e", help="epoch. default 1000.", default=1000, type=int)
parser.add_argument("-dv", help="device. default cpu.", default="cpu", type=str)
parser.add_argument("-hid", help="hidden size. default 128.", default=128, type=int)
parser.add_argument("-d", help="depth of networks. default 2.", default=2, type=int)
parser.add_argument("-cd", help="code dimension. default 2.", default=2, type=int)
parser.add_argument("-cnn", help="MLP (0) or CNN (1). default 0.", default=0, type=int)
parser.add_argument("-f", help="filters if CNN is used.", nargs="+", type=int)
parser.add_argument("-n", help="batch norm. default 0.", default=0, type=int)
parser.add_argument("-load", help="load model.", type=str)
parser.add_argument("-save", help="save model.", type=str, required=True)
args = parser.parse_args()

if not os.path.exists(args.save):
    os.makedirs(args.save)

arg_dict = vars(args)
for key in arg_dict.keys():
    print("%s: %s" % (key, arg_dict[key]))
    print("%s: %s" % (key, arg_dict[key]), file=open(os.path.join(args.save, "args.txt"), "a"))
print("date: %s" % time.asctime(time.localtime(time.time())))
print("date: %s" % time.asctime(time.localtime(time.time())), file=(open(os.path.join(args.save, "args.txt"), "a")))

device = torch.device(args.dv)
SIZE = 64

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.1120], [0.0197])
])
trainset = data.FirstLevelDataset(transform=transform)
loader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True)

normalization = "batch_norm" if args.n == 1 else None

if args.cnn == 0:
    encoder = torch.nn.Sequential(
        models.Flatten([1, 2, 3]),
        models.MLP([SIZE**2]+[args.hid]*args.d+[args.cd], normalization=normalization),
        models.STLayer()
    ).to(device)
else:
    L = len(args.f)-1
    denum = 4**L
    lat = args.f[-1] * ((SIZE // denum)**2)
    encoder = []
    bn = True if args.n == 1 else False
    for i in range(L):
        encoder.append(models.ConvBlock(args.f[i], args.f[i+1], 4, 1, 1, batch_norm=bn))
        encoder.append(models.ConvBlock(args.f[i+1], args.f[i+1], 4, 4, 1, batch_norm=bn))
    encoder.append(models.Flatten([1, 2, 3]))
    encoder.append(models.MLP([lat, args.cd]))
    encoder.append(models.STLayer())
    encoder = torch.nn.Sequential(*encoder).to(device)

decoder = models.MLP([args.cd + 3] + [args.hid] * args.d + [3]).to(device)
if args.load is not None:
    encoder.load_state_dict(torch.load(os.path.join(args.load, "encoder_first.ckpt")))
    decoder.load_state_dict(torch.load(os.path.join(args.load, "decoder_first.ckpt")))

print("="*10+"ENCODER"+"="*10)
print(encoder)
print("parameter count: %d" % utils.get_parameter_count(encoder))
print("="*27)
print("="*10+"DECODER"+"="*10)
print(decoder)
print("parameter count: %d" % utils.get_parameter_count(decoder))
print("="*27)

optimizer = torch.optim.Adam(
    lr=args.lr,
    params=[
        {"params": encoder.parameters()},
        {"params": decoder.parameters()}
    ],
    amsgrad=True
)
criterion = torch.nn.MSELoss(reduction="mean")
avg_loss = 0.0
it = 0
for e in range(args.e):
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
        print("it: %d, loss: %.4f" % (it, avg_loss/it))

with torch.no_grad():
    encoder.eval()
    x = trainset.objects
    if trainset.transform:
        x_t = []
        for x_i in x:
            x_t.append(trainset.transform(x_i))
        x = torch.stack(x_t, dim=0)
    codes = encoder(x.to(device)).cpu()
torch.save(codes, os.path.join(args.save, "codes_first.torch"))
torch.save(encoder.eval().cpu().state_dict(), os.path.join(args.save, "encoder_first.ckpt"))
torch.save(decoder.eval().cpu().state_dict(), os.path.join(args.save, "decoder_first.ckpt"))
