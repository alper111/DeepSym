import torch
import numpy as np


class FirstLevelDataset(torch.utils.data.Dataset):
    """ custom pytorch dataset class for first level object-action-effect data"""
    def __init__(self):
        self.action_names = np.load("data/action_names.npy")
        self.obj_names = np.load("data/obj_names.npy")

        self.objects = torch.load("data/object_depths.torch")
        self.targets = torch.load("data/targets.torch")
        self.actions = torch.load("data/actions.torch")
        self.effects = torch.load("data/effects_1.torch")

        # normalize the data
        self.objects = self.objects.reshape(-1, 128*128)
        self.obj_mu = self.objects.mean(dim=0)
        self.obj_std = self.objects.std(dim=0)
        self.objects = (self.objects - self.obj_mu) / (self.obj_std + 1e-6)
        self.objects = self.objects.reshape(-1, 1, 128, 128)

        self.effects = torch.cat([self.effects[:, :2], self.effects[:, 3:]], dim=1)
        self.eff_mu = self.effects.mean(dim=0)
        self.eff_std = self.effects.std(dim=0)
        self.effects = (self.effects - self.eff_mu) / (self.eff_std + 1e-6)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        sample = {}
        sample["object"] = self.objects[self.targets[idx]]
        sample["action"] = self.actions[idx]
        sample["effect"] = self.effects[idx]
        return sample


class SecondLevelDataset(torch.utils.data.Dataset):
    """ custom pytorch dataset class for second level object-action-effect data"""
    def __init__(self):
        self.action_names = np.load("data/action_names.npy")
        self.obj_names = np.load("data/obj_names.npy")

        self.objects = torch.load("data/object_depths.torch")
        self.codes = torch.load("data/codes_first.torch")
        self.relations = torch.load("data/relations.torch")
        self.effects = torch.load("data/effects_2.torch")

        # normalize the data
        self.objects = self.objects.reshape(-1, 128*128)
        self.obj_mu = self.objects.mean(dim=0)
        self.obj_std = self.objects.std(dim=0)
        self.objects = (self.objects - self.obj_mu) / (self.obj_std + 1e-6)
        self.objects = self.objects.reshape(-1, 1, 128, 128)

        self.eff_mu = self.effects.mean(dim=0)
        self.eff_std = self.effects.std(dim=0)
        self.effects = (self.effects - self.eff_mu) / (self.eff_std + 1e-6)
        self.effects = self.effects.abs()

    def __len__(self):
        return len(self.relations)

    def __getitem__(self, idx):
        sample = {}
        sample["object"] = self.objects[self.relations[idx]].squeeze(1)
        sample["code"] = self.codes[self.relations[idx]].reshape(-1)
        sample["effect"] = self.effects[idx]
        return sample
