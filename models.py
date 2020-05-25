import torch
import math
import os
import utils


class AffordanceModel:

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


class StraightThrough(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad):
        return grad.clamp(-1., 1.)


class STLayer(torch.nn.Module):

    def __init__(self):
        super(STLayer, self).__init__()
        self.func = StraightThrough.apply

    def forward(self, x):
        return self.func(x)


class Linear(torch.nn.Module):
    """ linear layer with optional batch normalization. """
    def __init__(self, in_features, out_features, std=None, batch_norm=False, gain=None):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        if batch_norm:
            self.batch_norm = torch.nn.BatchNorm1d(num_features=self.out_features)

        if std is not None:
            self.weight.data.normal_(0., std)
            self.bias.data.normal_(0., std)
        else:
            # defaults to linear activation
            if gain is None:
                gain = 1
            stdv = math.sqrt(gain / self.weight.size(1))
            self.weight.data.normal_(0., stdv)
            self.bias.data.zero_()

    def forward(self, x):
        x = torch.nn.functional.linear(x, self.weight, self.bias)
        if hasattr(self, "batch_norm"):
            x = self.batch_norm(x)
        return x

    def extra_repr(self):
        return "in_features={}, out_features={}".format(self.in_features, self.out_features)


class MLP(torch.nn.Module):
    """ multi-layer perceptron with batch norm option """
    def __init__(self, layer_info, activation=torch.nn.ReLU(), std=None, batch_norm=False, indrop=None, hiddrop=None):
        super(MLP, self).__init__()
        layers = []
        in_dim = layer_info[0]
        for i, unit in enumerate(layer_info[1:-1]):
            if i == 0 and indrop:
                layers.append(torch.nn.Dropout(indrop))
            elif i > 0 and hiddrop:
                layers.append(torch.nn.Dropout(hiddrop))
            layers.append(Linear(in_features=in_dim, out_features=unit, std=std, batch_norm=batch_norm, gain=2))
            layers.append(activation)
            in_dim = unit
        layers.append(Linear(in_features=in_dim, out_features=layer_info[-1], batch_norm=False))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def load(self, path, name):
        state_dict = torch.load(os.path.join(path, name+".ckpt"))
        self.load_state_dict(state_dict)

    def save(self, path, name):
        dv = self.layers[-1].weight.device
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.cpu().state_dict(), os.path.join(path, name+".ckpt"))
        self.train().to(dv)


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, std=None, bias=True,
                 batch_norm=False):
        super(ConvBlock, self).__init__()
        self.block = [torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, bias=bias)]
        if batch_norm:
            self.block.append(torch.nn.BatchNorm2d(out_channels))
        self.block.append(torch.nn.ReLU())

        if std is not None:
            self.block[0].weight.data.normal_(0., std)
            self.block[0].bias.data.normal_(0., std)
        self.block = torch.nn.Sequential(*self.block)

    def forward(self, x):
        return self.block(x)


class Flatten(torch.nn.Module):
    def __init__(self, dims):
        super(Flatten, self).__init__()
        self.dims = dims

    def forward(self, x):
        dim = 1
        for d in self.dims:
            dim *= x.shape[d]
        return x.reshape(-1, dim)

    def extra_repr(self):
        return "dims=[" + ", ".join(list(map(str, self.dims))) + "]"


class Avg(torch.nn.Module):
    def __init__(self, dims):
        super(Avg, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.mean(dim=self.dims)

    def extra_repr(self):
        return "dims=[" + ", ".join(list(map(str, self.dims))) + "]"


def build_encoder(opts, level):
    if level == 1:
        code_dim = opts["code1_dim"]
    else:
        code_dim = opts["code2_dim"]
    if opts["cnn"]:
        L = len(opts["filters"+str(level)])-1
        stride = 2
        encoder = []
        for i in range(L):
            encoder.append(ConvBlock(in_channels=opts["filters"+str(level)][i],
                                     out_channels=opts["filters"+str(level)][i+1],
                                     kernel_size=3, stride=1, padding=1, batch_norm=opts["batch_norm"]))
            encoder.append(ConvBlock(in_channels=opts["filters"+str(level)][i+1],
                                     out_channels=opts["filters"+str(level)][i+1],
                                     kernel_size=3, stride=stride, padding=1, batch_norm=opts["batch_norm"]))
        encoder.append(Avg([2, 3]))
        encoder.append(MLP([opts["filters"+str(level)][-1], code_dim]))
        encoder.append(STLayer())
    else:
        encoder = [
            Flatten([1, 2, 3]),
            MLP([[opts["size"]**2] + [opts["hidden_dim"]]*opts["depth"] + [code_dim]],
                batch_norm=opts["batch_norm"]),
            STLayer()]

    encoder = torch.nn.Sequential(*encoder)
    return encoder
