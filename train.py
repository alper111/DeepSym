import os
import time
import yaml
import torch
import models
import data


opts = yaml.safe_load(open("opts.yaml", "r"))
if not os.path.exists(opts["save"]):
    os.makedirs(opts["save"])
opts["time"] = time.asctime(time.localtime(time.time()))
file = open(os.path.join(opts["save"], "opts.yaml"), "w")
yaml.dump(opts, file)
file.close()
print(yaml.dump(opts))

device = torch.device(opts["device"])

transform = data.default_transform(size=opts["size"], affine=True, mean=0.279, std=0.0094)
trainset = data.FirstLevelDataset(transform=transform)
loader = torch.utils.data.DataLoader(trainset, batch_size=opts["batch_size"], shuffle=True)
load_all = torch.utils.data.DataLoader(trainset, batch_size=150, shuffle=False)

model = models.AffordanceModel(opts)
if opts["load"] is not None:
    model.load(opts["load"], ext="")
model.print_model()
model.train(opts["epoch"], loader)

sample = iter(load_all).next()
with torch.no_grad():
    codes = model.encoder(sample["object"].to(device)).cpu()

torch.save(codes, os.path.join(opts["save"], "codes_first.pt"))
