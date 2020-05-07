import torch
from torchvision import transforms
import numpy as np


class FirstLevelDataset(torch.utils.data.Dataset):
    """ custom pytorch dataset class for first level object-action-effect data """
    def __init__(self, transform=None):
        self.transform = transform
        self.action_names = np.load("data/action_names.npy")
        self.obj_names = np.load("data/obj_names.npy")

        self.objects = torch.load("data/objectsY.torch")
        self.targets = torch.load("data/targets.torch")
        self.actions = torch.load("data/actions.torch")
        self.effects = torch.load("data/effects_1.torch")

        self.effects = torch.cat([self.effects[:, :2], self.effects[:, 3:]], dim=1)
        self.eff_mu = self.effects.mean(dim=0)
        self.eff_std = self.effects.std(dim=0)
        self.effects = (self.effects - self.eff_mu) / (self.eff_std + 1e-6)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        sample = {}
        # sample["object"] = self.objects[self.targets[idx], np.random.randint(0, 25)]
        sample["object"] = self.objects[self.targets[idx], 12]
        sample["object"].unsqueeze_(0)
        if self.transform:
            sample["object"] = self.transform(sample["object"])
        sample["action"] = self.actions[idx]
        sample["effect"] = self.effects[idx]
        return sample


class SecondLevelDataset(torch.utils.data.Dataset):
    """ custom pytorch dataset class for second level object-action-effect data """
    def __init__(self, transform=None):
        self.transform = transform
        self.action_names = np.load("data/action_names.npy")
        self.obj_names = np.load("data/obj_names.npy")

        self.objects = torch.load("data/objectsY.torch")
        # self.codes = torch.load("save/codes_first.torch")
        self.relations = torch.load("data/relations.torch")
        self.effects = torch.load("data/effects_2.torch")

        self.effects = self.effects.abs()
        self.eff_mu = self.effects.mean(dim=0)
        self.eff_std = self.effects.std(dim=0)
        self.effects = (self.effects - self.eff_mu) / (self.eff_std + 1e-6)

    def __len__(self):
        return len(self.relations)

    def __getitem__(self, idx):
        sample = {}
        x = self.objects[self.relations[idx], np.random.randint(0, 25, (2,))]
        if self.transform:
            sample["object"] = torch.cat([self.transform(x[0]), self.transform(x[1])])
        else:
            sample["object"] = x
        # sample["code"] = self.codes[self.relations[idx]].reshape(-1)
        sample["effect"] = self.effects[idx]
        return sample


def default_transform(size, affine, mean, std):
    transform = [transforms.ToPILImage()]
    if size:
        transform.append(transforms.Resize(size))
    if affine:
        transform.append(
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                fillcolor=int(0.285*255)
            )
        )
    transform.append(transforms.ToTensor())
    if mean is not None:
        transform.append(transforms.Normalize([mean], [std]))
    transform = transforms.Compose(transform)
    return transform
