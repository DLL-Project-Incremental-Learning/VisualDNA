# src/vdna/networks/cityscapes_resnet101.py
# Adapted from https://github.com/facebookresearch/swav/blob/main/src/resnet50.py

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from ..utils.io import load_dict
from ..utils.settings import ExtractionSettings, NetworkSettings
from .feature_extraction_model import FeatureExtractionModel


def make_feature_dict(feature_list, preprend_name):
    feature_dict = {}
    for i in range(len(feature_list)):
        feature_dict[preprend_name + str(i)] = feature_list[i]
    return feature_dict


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        dont_return_features=False,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dont_return_features = dont_return_features

    def forward(self, x):
        if torch.is_tensor(x):
            features = []
        else:
            features = x[1]
            x = x[0]
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        features.append(out)

        if self.dont_return_features:
            return out
        else:
            return (out, features)


class ResNetFeatureExtractor(FeatureExtractionModel):
    def __init__(
        self,
        block,
        layers,
        zero_init_residual=False,
        groups=1,
        widen=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        normalize=False,
        output_dim=0,
        hidden_mlp=0,
        nmb_prototypes=0,
        eval_mode=False,
        dont_return_features=False,
        extraction_settings=ExtractionSettings(),
    ):
        # activation_ranges = hf_hub_download(
        #     repo_id=extraction_settings.hub_repo, filename="Random/rand_resnet50/activation_ranges.json"
        # )
        # min_max_act_per_neuron = load_dict(activation_ranges)
        # TODO: load activation ranges from a file
        filepath = "/media/henry/Data/msc/lab/proj/vdna/activation_ranges_cityscapes_resnet101.json"
        # min_max_act_per_neuron = load_dict("/media/henry/Data/Downloads/Random_rand_resnet101_activation_ranges.json")
        min_max_act_per_neuron = load_dict(filepath)

        network_settings = NetworkSettings(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
            {
                "layer1_0": 256,
                "layer1_1": 256,
                "layer1_2": 256,
                "layer2_0": 512,
                "layer2_1": 512,
                "layer2_2": 512,
                "layer2_3": 512,
                "layer3_0": 1024,
                "layer3_1": 1024,
                "layer3_2": 1024,
                "layer3_3": 1024,
                "layer3_4": 1024,
                "layer3_5": 1024,
                "layer3_6": 1024,
                "layer3_7": 1024,
                "layer3_8": 1024,
                "layer3_9": 1024,
                "layer3_10": 1024,
                "layer3_11": 1024,
                "layer3_12": 1024,
                "layer3_13": 1024,
                "layer3_14": 1024,
                "layer3_15": 1024,
                "layer3_16": 1024,
                "layer3_17": 1024,
                "layer3_18": 1024,
                "layer3_19": 1024,
                "layer3_20": 1024,
                "layer3_21": 1024,
                "layer3_22": 1024,
                "layer4_0": 2048,
                "layer4_1": 2048,
                "layer4_2": 2048,
            },
            (224, 224),
            "cityscapes_resnet101_backbone",
        )

        super(ResNetFeatureExtractor, self).__init__(network_settings, extraction_settings, min_max_act_per_neuron)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.eval_mode = eval_mode
        self.padding = nn.ConstantPad2d(1, 0.0)

        self.inplanes = width_per_group * widen
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # change padding 3 -> 2 compared to original torchvision code because added a padding layer
        num_out_filters = width_per_group * widen
        self.conv1 = nn.Conv2d(3, num_out_filters, kernel_size=7, stride=2, padding=2, bias=False)
        self.bn1 = norm_layer(num_out_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, num_out_filters, layers[0], dont_return_features=dont_return_features)
        num_out_filters *= 2
        self.layer2 = self._make_layer(
            block,
            num_out_filters,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            dont_return_features=dont_return_features,
        )
        num_out_filters *= 2
        self.layer3 = self._make_layer(
            block,
            num_out_filters,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            dont_return_features=dont_return_features,
        )
        num_out_filters *= 2
        self.layer4 = self._make_layer(
            block,
            num_out_filters,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            dont_return_features=dont_return_features,
        )
        # normalize output features
        self.l2norm = normalize

        # projection head
        if output_dim == 0:
            self.projection_head = None
        elif hidden_mlp == 0:
            self.projection_head = nn.Linear(num_out_filters * block.expansion, output_dim)
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(num_out_filters * block.expansion, hidden_mlp),
                nn.BatchNorm1d(hidden_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_mlp, output_dim),
            )

        # prototype layer
        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(output_dim, nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        self.dont_return_features = dont_return_features

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, dont_return_features=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                dont_return_features,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    dont_return_features=dont_return_features,
                )
            )

        return nn.Sequential(*layers)

    def forward_backbone(self, x):
        x = self.padding(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if self.dont_return_features:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        else:
            x, features_layer1 = self.layer1(x)
            x, features_layer2 = self.layer2(x)
            x, features_layer3 = self.layer3(x)
            x, features_layer4 = self.layer4(x)

        if self.eval_mode:
            return x

        if self.dont_return_features:
            return x
        else:
            features = {
                **make_feature_dict(features_layer1, "layer1_"),
                **make_feature_dict(features_layer2, "layer2_"),
                **make_feature_dict(features_layer3, "layer3_"),
                **make_feature_dict(features_layer4, "layer4_"),
            }

            return x, features

    def forward_head(self, x):
        if self.projection_head is not None:
            x = self.projection_head(x)

        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)

        if self.prototypes is not None:
            return x, self.prototypes(x)
        return x

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in inputs]),
                return_counts=True,
            )[1],
            0,
        )
        start_idx = 0
        for end_idx in idx_crops:
            if self.dont_return_features:
                _out = self.forward_backbone(
                    torch.cat(inputs[start_idx:end_idx]).cuda(device=inputs[0].device, non_blocking=True),
                )
                features = {}
            else:
                _out, features = self.forward_backbone(
                    torch.cat(inputs[start_idx:end_idx]).cuda(device=inputs[0].device, non_blocking=True),
                )
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx

        # We don't use a head
        # features["head"] = self.forward_head(output)

        return features

    def get_features(self, batch):
        return self.forward(batch)

    def load_state_dict_with_key_update(self, state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("backbone."):
                new_k = k[len("backbone."):]
                new_state_dict[new_k] = v
        self.load_state_dict(new_state_dict)

class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module("prototypes" + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out


def resnet101_feat_extractor(extraction_settings):
    resnet = ResNetFeatureExtractor(Bottleneck, [3, 4, 23, 3], extraction_settings=extraction_settings)
    # weights = hf_hub_download(repo_id=extraction_settings.hub_repo, filename="Random/rand_resnet50/weights.pth")
    weights_path = "/media/henry/Data/Downloads/best_deeplabv3plus_resnet101_cityscapes_os16.pth"
    checkpoint = torch.load(weights_path)
    # resnet.load_state_dict(checkpoint["model_state"])
    resnet.load_state_dict_with_key_update(checkpoint["model_state"])
    # for name, param in resnet.named_parameters():
        # print(f"name: {name}")
        # if "layer1" in name:
            # print(f"name: {name}")
            # print(f"param: {param}")
    return resnet
