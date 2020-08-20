import torch
from torchvision import transforms
import numpy as np


class ImageFirstLevel(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.observation = torch.load("data/img/obs_prev_z.pt").unsqueeze(1)
        self.obs_mu = self.observation.mean()
        self.obs_std = self.observation.std()
        self.observation = (self.observation - self.obs_mu) / (self.obs_std + 1e-6)

        self.effect = torch.load("data/img/obs_next.pt").unsqueeze(1)
        self.eff_mu = self.effect.mean()
        self.eff_std = self.effect.std()
        self.effect = (self.effect - self.eff_mu) / (self.eff_std + 1e-6)

        self.action = torch.load("data/img/action.pt")
        self.force = torch.load("data/img/force.pt").relu()

    def __len__(self):
        return len(self.observation)

    def __getitem__(self, idx):
        sample = {}
        sample["observation"] = self.observation[idx]
        sample["effect"] = self.effect[idx]
        sample["action"] = self.action[idx]
        sample["force"] = self.force[idx]
        if self.transform:
            sample["observation"] = self.transform(self.observation[idx])
        return sample


class FirstLevelDataset(torch.utils.data.Dataset):
    """ custom pytorch dataset class for first level object-action-effect data """
    def __init__(self, transform=None):
        self.transform = transform
        self.action_names = np.load("data/action_names.npy")
        self.obj_names = np.load("data/obj_names.npy")

        self.objects = torch.load("data/objectsZ.pt")
        self.targets = torch.load("data/targets.pt")
        self.actions = torch.load("data/actions.pt")
        self.effects = torch.load("data/effects_1.pt")

        self.effects = torch.cat([self.effects[:, :2], self.effects[:, 3:]], dim=1)
        self.eff_mu = self.effects.mean(dim=0)
        self.eff_std = self.effects.std(dim=0)
        self.effects = (self.effects - self.eff_mu) / (self.eff_std + 1e-6)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        sample = {}
        sample["object"] = self.objects[self.targets[idx], np.random.randint(0, 25)]
        # sample["object"] = self.objects[self.targets[idx], 12]
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

        self.objects = torch.load("data/objectsZ.pt")
        self.relations = torch.load("data/relations.pt")
        self.effects = torch.load("data/effects_2.pt")

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
        sample["effect"] = self.effects[idx]
        return sample


def default_transform(size, affine, mean=None, std=None):
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
