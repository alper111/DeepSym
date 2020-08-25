import os
import argparse
import time
import yaml
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from models import EffectRegressorConv
import data
import utils

parser = argparse.ArgumentParser("Train effect prediction models.")
parser.add_argument("-opts", help="option file", type=str, required=True)
args = parser.parse_args()

opts = yaml.safe_load(open(args.opts, "r"))
if not os.path.exists(opts["save"]):
    os.makedirs(opts["save"])
opts["time"] = time.asctime(time.localtime(time.time()))
file = open(os.path.join(opts["save"], "opts.yaml"), "w")
yaml.dump(opts, file)
file.close()
print(yaml.dump(opts))

device = torch.device(opts["device"])

# load the first level data
transform = data.default_transform(size=42, affine=True)
# transform = None
trainset = data.ImageFirstLevel(transform=transform)
loader = torch.utils.data.DataLoader(trainset, batch_size=opts["batch_size"], shuffle=True)

model = EffectRegressorConv(opts)
if opts["load"] is not None:
    model.load(opts["load"], ext="")
model.print_model()
model.train(opts["epoch"], loader)

d = 2**opts["code_dim"]
x = -torch.ones(3*d, 3+opts["code_dim"], device=opts["device"])
for i in range(3):
    for j in range(d):
        x[i*d+j, :opts["code_dim"]] = torch.tensor(utils.decimal_to_binary(j, length=opts["code_dim"]), dtype=torch.float)
x[:d, opts["code_dim"]:] = torch.tensor([1., -1., -1.]).repeat(d, 1)
x[d:2*d, opts["code_dim"]:] = torch.tensor([-1., 1., -1.]).repeat(d, 1)
x[2*d:, opts["code_dim"]:] = torch.tensor([-1., -1., 1.]).repeat(d, 1)

with torch.no_grad():
    v, _ = model.decode(x)
    v = v * (trainset.obs_std + 1e-6) + trainset.obs_mu
    v = v.cpu()

fig, ax = plt.subplots(3, d, dpi=150)
for i in range(3):
    for j in range(d):
        ax[i][j].imshow(v[i*d+j, 0])
pp = PdfPages(os.path.join(opts["save"], "decodings.pdf"))
pp.savefig(fig)
pp.close()
