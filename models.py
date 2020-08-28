import os
import torch
import utils
from blocks import MLP, build_encoder


class EffectRegressor:

    def __init__(self, opts):
        self.device = torch.device(opts["device"])
        self.encoder = build_encoder(opts, 1).to(self.device)
        layer_info = [opts["code1_dim"]+3] + [opts["hidden_dim"]] * opts["depth"] + [3]
        self.decoder = MLP(layer_info).to(self.device)
        self.optimizer = torch.optim.Adam(lr=opts["learning_rate"],
                                          params=[
                                              {"params": self.encoder.parameters()},
                                              {"params": self.decoder.parameters()}],
                                          amsgrad=True,
                                          weight_decay=opts["weight_decay"])
        self.criterion = torch.nn.MSELoss()
        self.iteration = 0
        self.save_path = opts["save"]

    def loss(self, sample):
        obs = sample["observation"].to(self.device)
        effect = sample["effect"].to(self.device)
        action = sample["action"].to(self.device)

        h = self.encoder(obs)
        h_aug = torch.cat([h, action], dim=-1)
        out = self.decoder(h_aug)
        loss = self.criterion(out, effect)
        return loss

    def one_pass_optimize(self, loader):
        running_avg_loss = 0.0
        for i, sample in enumerate(loader):
            self.optimizer.zero_grad()
            loss = self.loss(sample)
            loss.backward()
            running_avg_loss += loss.item()
            self.iteration += 1
            self.optimizer.step()
        return running_avg_loss/i

    def train(self, epoch, loader):
        best_loss = 1e100
        for e in range(epoch):
            epoch_loss = self.one_pass_optimize(loader)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                self.save(self.save_path, "_best")
            print("Epoch: %d, iter: %d, loss: %.4f" % (e+1, self.iteration, epoch_loss))
            self.save(self.save_path, "_last")

    def load(self, path, ext):
        encoder_dict = torch.load(os.path.join(path, "encoder"+ext+".ckpt"))
        decoder_dict = torch.load(os.path.join(path, "decoder"+ext+".ckpt"))
        self.encoder.load_state_dict(encoder_dict)
        self.decoder.load_state_dict(decoder_dict)

    def save(self, path, ext):
        encoder_dict = self.encoder.eval().cpu().state_dict()
        decoder_dict = self.decoder.eval().cpu().state_dict()
        torch.save(encoder_dict, os.path.join(path, "encoder"+ext+".ckpt"))
        torch.save(decoder_dict, os.path.join(path, "decoder"+ext+".ckpt"))
        self.encoder.train().to(self.device)
        self.decoder.train().to(self.device)

    def print_model(self):
        print("="*10+"ENCODER"+"="*10)
        print(self.encoder)
        print("parameter count: %d" % utils.get_parameter_count(self.encoder))
        print("="*27)
        print("="*10+"DECODER"+"="*10)
        print(self.decoder)
        print("parameter count: %d" % utils.get_parameter_count(self.decoder))
        print("="*27)


class EffectRegressorMLP:

    def __init__(self, opts):
        self.device = torch.device(opts["device"])
        self.encoder1 = build_encoder(opts, 1).to(self.device)
        self.encoder2 = build_encoder(opts, 2).to(self.device)
        self.decoder1 = MLP([opts["code1_dim"] + 3] + [opts["hidden_dim"]] * opts["depth"] + [3]).to(self.device)
        self.decoder2 = MLP([opts["code2_dim"] + opts["code1_dim"]*2] + [opts["hidden_dim"]] * opts["depth"] + [6]).to(self.device)
        self.optimizer = torch.optim.Adam(lr=opts["learning_rate"],
                                          params=[
                                              {"params": self.encoder1.parameters()},
                                              {"params": self.encoder2.parameters()},
                                              {"params": self.decoder1.parameters()},
                                              {"params": self.decoder2.parameters()}],
                                          amsgrad=True)
        self.criterion = torch.nn.MSELoss()
        self.iteration = 0
        self.save_path = opts["save"]

    def loss1(self, sample):
        h = self.encoder1(sample["object"].to(self.device))
        action = torch.eye(3, device=self.device)[sample["action"]]
        h_aug = torch.cat([h, action], dim=-1)
        effect_pred = self.decoder1(h_aug)
        loss = self.criterion(effect_pred, sample["effect"].to(self.device))
        return loss

    def loss2(self, sample):
        obj = sample["object"].to(self.device)
        with torch.no_grad():
            h1 = self.encoder1(obj.reshape(-1, 1, obj.shape[2], obj.shape[3]))
        h1 = h1.reshape(obj.shape[0], -1)
        h2 = self.encoder2(obj)
        h_aug = torch.cat([h1, h2], dim=-1)
        effect_pred = self.decoder2(h_aug)
        loss = self.criterion(effect_pred, sample["effect"].to(self.device))
        return loss

    def one_pass_optimize(self, loader, level):
        running_avg_loss = 0.0
        for i, sample in enumerate(loader):
            self.optimizer.zero_grad()
            if level == 1:
                loss = self.loss1(sample)
            else:
                loss = self.loss2(sample)
            loss.backward()
            running_avg_loss += loss.item()
            self.iteration += 1
            self.optimizer.step()
        return running_avg_loss/i

    def train(self, epoch, loader, level):
        best_loss = 1e100
        for e in range(epoch):
            epoch_loss = self.one_pass_optimize(loader, level)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                self.save(self.save_path, "_best", level)
            print("Epoch: %d, iter: %d, loss: %.4f" % (e+1, self.iteration, epoch_loss))
            self.save(self.save_path, "_last", level)

    def load(self, path, ext, level):
        if level == 1:
            encoder = self.encoder1
            decoder = self.decoder1
        else:
            encoder = self.encoder2
            decoder = self.decoder2

        encoder_dict = torch.load(os.path.join(path, "encoder"+str(level)+ext+".ckpt"))
        decoder_dict = torch.load(os.path.join(path, "decoder"+str(level)+ext+".ckpt"))
        encoder.load_state_dict(encoder_dict)
        decoder.load_state_dict(decoder_dict)

    def save(self, path, ext, level):
        if level == 1:
            encoder = self.encoder1
            decoder = self.decoder1
        else:
            encoder = self.encoder2
            decoder = self.decoder2

        encoder_dict = encoder.eval().cpu().state_dict()
        decoder_dict = decoder.eval().cpu().state_dict()
        torch.save(encoder_dict, os.path.join(path, "encoder"+str(level)+ext+".ckpt"))
        torch.save(decoder_dict, os.path.join(path, "decoder"+str(level)+ext+".ckpt"))
        encoder.train().to(self.device)
        decoder.train().to(self.device)

    def print_model(self, level):
        encoder = self.encoder1 if level == 1 else self.encoder2
        decoder = self.decoder1 if level == 1 else self.decoder2
        print("="*10+"ENCODER"+"="*10)
        print(encoder)
        print("parameter count: %d" % utils.get_parameter_count(encoder))
        print("="*27)
        print("="*10+"DECODER"+"="*10)
        print(decoder)
        print("parameter count: %d" % utils.get_parameter_count(decoder))
        print("="*27)
