import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim, hidden_dim, activation=nn.ReLU, dropout=0.0):
        super().__init__()
        layers = [nn.Linear(dim, hidden_dim), activation()]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, dim))
        self.fc = nn.Sequential(*layers)
        self.activation = activation()

    def forward(self, x):
        return self.activation(x + self.fc(x))


class RankNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_blocks=3, activation=nn.ReLU, dropout=0.0):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim, activation, dropout)
            for _ in range(num_blocks)
        ])
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        x = torch.relu(self.input_layer(x))
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x).squeeze(-1)


class FeatureExtractor(nn.Module):
    def __init__(self, dim=1280, hidden=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim)
        )

    def forward(self, x):
        return x + self.net(x)


class DomainClassifier(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_=1.0):
    return GradientReversalFunction.apply(x, lambda_)


class GRL(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return grad_reverse(x, self.lambda_)
