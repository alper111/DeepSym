import torch
import math


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
    '''linear layer with optional batch normalization or layer normalization'''
    def __init__(self, in_features, out_features, std=None, normalization=None):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        self.normalization = normalization
        if normalization == 'batch_norm':
            self.normalization_func = torch.nn.BatchNorm1d(num_features=self.out_features)
        elif normalization == 'layer_norm':
            self.normalization_func = torch.nn.LayerNorm(normalized_shape=self.out_features)

        if std is not None:
            self.weight.data.normal_(0., std)
            self.bias.data.normal_(0., std)
        else:
            # he initialization for ReLU activation
            stdv = math.sqrt(2 / self.weight.size(1))
            self.weight.data.normal_(0., stdv)
            self.bias.data.zero_()

    def forward(self, x):
        x = torch.nn.functional.linear(x, self.weight, self.bias)
        if self.normalization:
            x = self.normalization_func(x)
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, normalization={}'.format(
            self.in_features, self.out_features, self.normalization
        )


class MLP(torch.nn.Module):
    '''multi-layer perceptron with batch norm option'''
    def __init__(self, layer_info, activation=torch.nn.ReLU(), std=None, normalization=None):
        super(MLP, self).__init__()
        layers = []
        in_dim = layer_info[0]
        for l in layer_info[1:-1]:
            layers.append(Linear(in_features=in_dim, out_features=l, std=std, normalization=normalization))
            layers.append(activation)
            in_dim = l
        layers.append(Linear(in_features=in_dim, out_features=layer_info[-1], std=std, normalization=None))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


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
