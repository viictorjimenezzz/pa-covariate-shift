import os
import hydra

import pandas as pd

import torchvision
from torch import nn
import torch


class DGBackbone(nn.Module):
    def __init__(
        self,
        n_classes: int,
        net: str = "resnet50",
        pretrained: bool = True,
        input_channels: int = 3,
    ):
        super(DGBackbone, self).__init__()

        if "resnet" in net:
            if str(18) in net:
                self.net = torchvision.models.resnet18(True)
            elif str(34) in net:
                self.net = torchvision.models.resnet34(True)
            elif str(50) in net:
                self.net = torchvision.models.resnet50(True)
            elif str(101) in net:
                self.net = torchvision.models.resnet101(True)

            if input_channels != 3:
                self.net.conv1 = nn.Conv2d(
                    input_channels,
                    64,
                    kernel_size=7,
                    stride=1,
                    padding=3,
                    bias=False,
                )
                self.net.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            self.net.fc = nn.Sequential(
                nn.Linear(self.net.fc.in_features, 500),
                nn.BatchNorm1d(500),
                nn.Dropout(0.2),
                nn.Linear(500, 256),
                nn.Linear(256, n_classes),
            )

        elif "densenet" in net:
            if str(121) in net:
                self.net = torchvision.models.densenet121(pretrained=pretrained)

            if input_channels != 3:
                self.net.features.conv0 = nn.Conv2d(
                    1,
                    64,
                    kernel_size=(7, 7),
                    stride=(2, 2),
                    padding=(3, 3),
                    bias=False,
                )

            self.net.classifier = nn.Sequential(
                nn.Linear(self.net.classifier.in_features, 500),
                nn.BatchNorm1d(500),
                nn.Dropout(0.2),
                nn.Linear(500, 256),
                nn.Linear(256, n_classes),
            )

        elif "efficient" in net:
            self.net = eval(
                "torchvision.models.efficientnet_"
                + net[-2:]
                + "(pretrained="
                + str(pretrained)
                + ")"
            )
            if input_channels != 3:
                out_features = self.net.features[0][0].out_channels
                self.net.features[0][0] = nn.Conv2d(
                    1,
                    out_features,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                    bias=False,
                )

            self.net.classifier = nn.Sequential(
                nn.Linear(self.net.classifier[1].in_features, 500),
                nn.BatchNorm1d(500),
                nn.Dropout(0.2),
                nn.Linear(500, 256),
                nn.Linear(256, n_classes),
            )

    def forward(self, x):
        h = self.net(x)

        return h


def get_lm_model(exp_name: str, net: nn.Module, log_dir: str) -> nn.Module:
    ckpt_path = os.path.join(log_dir, "ckpt_exp.csv")
    df_ckpt = pd.read_csv(ckpt_path)
    ckpt_path = df_ckpt[
        df_ckpt["experiment_name"] == exp_name
    ].ckpt_path.values[0]

    ckpt_lightning = torch.load(ckpt_path)
    weights = ckpt_lightning["state_dict"].copy()
    for key in ckpt_lightning["state_dict"].keys():
        if "model" in key:
            weights[key[6:]] = weights.pop(key)

    net.load_state_dict(weights)

    return net
