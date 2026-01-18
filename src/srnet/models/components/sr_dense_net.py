import torch
import torch.nn as nn


class ConvNode(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, padding):
        super().__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
        )
        self.add_module("relu", nn.ReLU(inplace=True))
        # self.add_module('norm', nn.BatchNorm2d(output_channels))
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain("relu"))

    def forward(self, input):
        return self.relu(self.conv(input))  # self.norm(self.relu(self.conv(input)))


class DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, channels, kernel_size, padding):
        super().__init__()
        for i in range(num_layers):
            layer = ConvNode(channels * (i + 1), channels, kernel_size, padding)
            self.add_module("conv%d" % i, layer)
        self.add_module(
            "last_conv",
            ConvNode((num_layers + 1) * channels, channels, kernel_size=1, padding=0),
        )

    def forward(self, input):
        features = input
        for name, layer in self.items():
            if name[:4] == "conv":
                new_features = layer(features)
                features = torch.cat([features, new_features], 1)
        features = self.last_conv(features)
        return features


class GlobalDenseBlock(nn.ModuleDict):
    def __init__(self, num_dense_blocks, num_layers, channels, kernel_size, padding):
        super().__init__()
        for i in range(num_dense_blocks):
            rdb = DenseBlock(num_layers, channels, kernel_size, padding)
            self.add_module("rdb%d" % i, rdb)
        self.add_module(
            "last_conv",
            ConvNode(num_dense_blocks * channels, channels, kernel_size=1, padding=0),
        )

    def forward(self, input):
        features = input
        all_features = None
        for name, layer in self.items():
            if name[:3] == "rdb":
                features = layer(features)
                if all_features is not None:
                    all_features = torch.cat([all_features, features], 1)
                else:
                    all_features = features
        all_features = self.last_conv(all_features)
        return all_features


class DenseNet(nn.ModuleDict):
    def __init__(
        self,
        num_dense_blocks=4,
        num_layers=3,
        channels=64,
        kernel_size=3,
        padding=1,
        upscale_factor=4,
    ):
        super().__init__()
        self.add_module("conv1", ConvNode(3, channels, kernel_size=3, padding=1))
        self.add_module("conv2", ConvNode(channels, channels, kernel_size=3, padding=1))
        self.add_module(
            "global_dense",
            GlobalDenseBlock(num_dense_blocks, num_layers, channels, kernel_size, padding),
        )
        self.add_module("conv3", ConvNode(channels, channels, kernel_size=3, padding=1))
        self.add_module("upscale", nn.PixelShuffle(upscale_factor))
        self.add_module(
            "conv4",
            ConvNode(
                channels // upscale_factor**2,
                3,
                kernel_size=1,
                padding=0,
            ),
        )

    def forward(self, input):
        input = self.conv1(input)
        features = self.conv2(input)
        features = self.global_dense(features)
        features = self.conv3(features) + input
        features = self.upscale(features)
        features = self.conv4(features)
        return features


if __name__ == "__main__":
    _ = DenseNet()
