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

# load the first level data
transform = data.default_transform(size=opts["size"], affine=True, mean=0.279, std=0.0094)
trainset = data.FirstLevelDataset(transform=transform)
loader = torch.utils.data.DataLoader(trainset, batch_size=opts["batch_size"], shuffle=True)

model = models.AffordanceModel(opts)
if opts["load"] is not None:
    model.load(opts["load"], ext="", level=1)
    model.load(opts["load"], ext="", level=2)
model.print_model(1)
model.train(opts["epoch"], loader, 1)

# load the best encoder1
model.load(opts["save"], "_best", 1)

# change hyperparams for the second level
opts["batch_size"] = 50
opts["epoch"] = 200

# load the second level data
transform = data.default_transform(size=opts["size"], affine=True, mean=0.279, std=0.0094)
trainset = data.SecondLevelDataset(transform=transform)
loader = torch.utils.data.DataLoader(trainset, batch_size=opts["batch_size"], shuffle=True)
model.print_model(2)
model.train(opts["epoch"], loader, 2)
