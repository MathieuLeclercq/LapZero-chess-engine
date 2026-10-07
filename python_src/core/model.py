
import sys as _runtime_sys
from pathlib import Path as _RuntimePath
if __package__ in (None, ""):
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
import _runtime

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Mécanisme Squeeze-and-Excitation pour recalibrer les filtres."""
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        # Le 'squeeze' réduit la dimension spatiale 8x8 à 1x1
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        # L''excitation' apprend quels filtres allumer ou éteindre
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y  # On multiplie les canaux originaux par leur score d'importance


class ResBlock(nn.Module):
    """Bloc résiduel intégrant le Squeeze-and-Excitation."""
    def __init__(self, num_filters):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.se = SEBlock(num_filters)  # Ajout du bloc SE

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)  # Application du SE avant l'addition au résidu
        out += residual
        return F.relu(out)


class ChessNet(nn.Module):
    def __init__(self, num_res_blocks=10, num_filters=128):
        super(ChessNet, self).__init__()

        # 1. Couche d'entrée
        self.conv_input = nn.Conv2d(119, num_filters, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_filters)

        # 2. Corps résiduel
        self.res_blocks = nn.ModuleList([ResBlock(num_filters) for _ in range(num_res_blocks)])

        # 3. Tête de Policy (Optimisée avec un Bottleneck)
        policy_bottleneck_channels = 32  # AlphaZero classique utilise 2 canaux, Lc0 entre 32 et 80.

        self.policy_conv1 = nn.Conv2d(num_filters, policy_bottleneck_channels, kernel_size=1)
        self.policy_bn1 = nn.BatchNorm2d(policy_bottleneck_channels)
        self.policy_conv2 = nn.Conv2d(policy_bottleneck_channels, 73, kernel_size=1)

        # 4. Tête de Value (Élargie à 32 filtres comme vu précédemment)
        self.value_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Initialisation "Tabula Rasa" pour éviter l'explosion de la Loss
        nn.init.zeros_(self.value_fc2.weight)
        nn.init.zeros_(self.value_fc2.bias)
        nn.init.zeros_(self.policy_conv2.weight)
        nn.init.zeros_(self.policy_conv2.bias)

    def forward(self, x):
        # x shape: (N, 119, 8, 8)
        x = F.relu(self.bn_input(self.conv_input(x)))

        for block in self.res_blocks:
            x = block(x)

        # Policy Head
        p = F.relu(self.policy_bn1(self.policy_conv1(x)))
        p = self.policy_conv2(p)  # Shape: (N, 73, 8, 8)
        p = p.view(p.size(0), -1)  # Flatten final en (N, 4672)

        # Value Head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))  # Sortie entre -1 et 1

        return p, v
