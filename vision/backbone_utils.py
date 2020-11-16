from collections import OrderedDict
from torch import nn
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
import torch.nn.functional as F
from torchvision.ops import misc as misc_nn_ops
from ._utils import IntermediateLayerGetter
from . import resnet
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3
from torchvision.models.segmentation.fcn import FCN
import torch


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels, kernel_size=3, padding=1, last=True, is_real=False):
        inter_channels = in_channels // 2
        # This one works for real robot as well. Due to version control issue, we ran one verison for real robot and one for sim. But it shouldn't matter much.
        if is_real:
            layers = [
                    nn.Conv2d(in_channels, inter_channels, kernel_size, padding=padding, bias=False),
                    nn.BatchNorm2d(inter_channels),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Conv2d(inter_channels, channels, 1)
            ]
        # Presented in the paper, but the above one also works.
        else:
            if last:
                layers = [
                    nn.Conv2d(in_channels, inter_channels, kernel_size, padding=padding, bias=False),
                    nn.BatchNorm2d(inter_channels),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Conv2d(inter_channels, channels, 1)
                ]
            else:
                layers = [
                    nn.Conv2d(in_channels, inter_channels, kernel_size, padding=padding, bias=False),
                    nn.BatchNorm2d(inter_channels),
                    nn.ReLU(),
                    nn.Dropout(0.1),  # this can be removed
                    nn.Conv2d(inter_channels, channels, 1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(),
                    nn.Dropout(0.1)  # this can be removed
                ]
        
        super(FCNHead, self).__init__(*layers)



class BackboneWithFPNAndHeadPush(nn.Module):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediatLayerGetter apply here.
    Arguments:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
    Attributes:
        out_channels (int): the number of channels in the FPN
    """
    def __init__(self, backbone, return_layers, in_channels_list, out_channels, is_real=False):
        super().__init__()
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )

        inplanes = 256  # the channels of 'out' layer.
        final_out_channels = 1
        self.classifier1 = FCNHead(inplanes, 64, last=False, is_real=is_real)
        self.classifier2 = FCNHead(64, final_out_channels, last=True, is_real=is_real)

        self.out_channels = out_channels

    def forward(self, x):
        input_shape_half = (x.shape[-2] // 2, x.shape[-1] // 2)
        input_shape = x.shape[-2:]

        x = self.body(x)
        x = self.fpn(x)
        x = x["0"]
        x = self.classifier1(x)
        x = F.interpolate(x, size=input_shape_half, mode='nearest')
        x = self.classifier2(x)
        x = F.interpolate(x, size=input_shape, mode='nearest')

        return x

class BackboneWithFPNAndHeadGrasp(nn.Module):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediatLayerGetter apply here.
    Arguments:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
    Attributes:
        out_channels (int): the number of channels in the FPN
    """
    def __init__(self, backbone, return_layers, in_channels_list, out_channels, is_real=False):
        super().__init__()
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )

        self.head = nn.Sequential(OrderedDict([
            ('head-conv0', nn.Conv2d(1, 1, kernel_size=(11, 57), padding=(5, 28), bias=False)),
            ('head-relu0', nn.ReLU(inplace=True)),
            ('head-conv1', nn.Conv2d(1, 1, kernel_size=(11, 57), padding=(5, 28), bias=False))
        ]))

        inplanes = 256  # the channels of 'out' layer.
        final_out_channels = 1
        self.classifier1 = FCNHead(inplanes, 64, last=False, is_real=is_real)
        self.classifier2 = FCNHead(64, final_out_channels, last=False, is_real=is_real)

        self.out_channels = out_channels

    def forward(self, x):
        input_shape_half = (x.shape[-2] // 2, x.shape[-1] // 2)
        input_shape = x.shape[-2:]

        x = self.body(x)
        x = self.fpn(x)
        x = x["0"]

        x = self.classifier1(x)
        x = F.interpolate(x, size=input_shape_half, mode='nearest')
        x = self.classifier2(x)
        x = F.interpolate(x, size=input_shape, mode='nearest')

        x = self.head(x)

        return x

class BackboneWithFPN(nn.Module):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediatLayerGetter apply here.
    Arguments:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
    Attributes:
        out_channels (int): the number of channels in the FPN
    """
    def __init__(self, backbone, return_layers, in_channels_list, out_channels):
        super(BackboneWithFPN, self).__init__()
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x

def resnet_fpn_net(backbone_name, norm_layer=misc_nn_ops.FrozenBatchNorm2d, trainable_layers=5, grasp=True, is_real=False):
    backbone = resnet.__dict__[backbone_name](
        pretrained=False,
        norm_layer=norm_layer,
        input_channels=4)
    """
    Constructs a specified ResNet backbone with FPN on top. Freezes the specified number of layers in the backbone.

    Examples::

        >>> from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
        >>> backbone = resnet_fpn_backbone('resnet50', pretrained=True, trainable_layers=3)
        >>> # get some dummy image
        >>> x = torch.rand(1,3,64,64)
        >>> # compute the output
        >>> output = backbone(x)
        >>> print([(k, v.shape) for k, v in output.items()])
        >>> # returns
        >>>   [('0', torch.Size([1, 256, 16, 16])),
        >>>    ('1', torch.Size([1, 256, 8, 8])),
        >>>    ('2', torch.Size([1, 256, 4, 4])),
        >>>    ('3', torch.Size([1, 256, 2, 2])),
        >>>    ('pool', torch.Size([1, 256, 1, 1]))]

    Arguments:
        backbone_name (string): resnet architecture. Possible values are 'ResNet', 'resnet18', 'resnet34', 'resnet50',
             'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
        norm_layer (torchvision.ops): it is recommended to use the default value. For details visit:
            (https://github.com/facebookresearch/maskrcnn-benchmark/issues/267)
        pretrained (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
    """
    # select layers that wont be frozen
    assert trainable_layers <= 5 and trainable_layers >= 0
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    # freeze layers only if pretrained backbone is used
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channels = 256
    if grasp:
        return BackboneWithFPNAndHeadGrasp(backbone, return_layers, in_channels_list, out_channels, is_real)
    else:
        return BackboneWithFPNAndHeadPush(backbone, return_layers, in_channels_list, out_channels, is_real)

def resent_backbone(backbone_name, pretrained, num_classes, input_channels, norm_layer=misc_nn_ops.FrozenBatchNorm2d):
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained,
        norm_layer=norm_layer,
        input_channels=input_channels,
        num_classes=num_classes)
    return backbone

def resnet_fpn_backbone(backbone_name, pretrained, norm_layer=misc_nn_ops.FrozenBatchNorm2d, trainable_layers=3):
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained,
        norm_layer=norm_layer)
    """
    Constructs a specified ResNet backbone with FPN on top. Freezes the specified number of layers in the backbone.
    Examples::
        >>> from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
        >>> backbone = resnet_fpn_backbone('resnet50', pretrained=True, trainable_layers=3)
        >>> # get some dummy image
        >>> x = torch.rand(1,3,64,64)
        >>> # compute the output
        >>> output = backbone(x)
        >>> print([(k, v.shape) for k, v in output.items()])
        >>> # returns
        >>>   [('0', torch.Size([1, 256, 16, 16])),
        >>>    ('1', torch.Size([1, 256, 8, 8])),
        >>>    ('2', torch.Size([1, 256, 4, 4])),
        >>>    ('3', torch.Size([1, 256, 2, 2])),
        >>>    ('pool', torch.Size([1, 256, 1, 1]))]
    Arguments:
        backbone_name (string): resnet architecture. Possible values are 'ResNet', 'resnet18', 'resnet34', 'resnet50',
             'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
        norm_layer (torchvision.ops): it is recommended to use the default value. For details visit:
            (https://github.com/facebookresearch/maskrcnn-benchmark/issues/267)
        pretrained (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
    """
    # select layers that wont be frozen
    assert trainable_layers <= 5 and trainable_layers >= 0
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    # freeze layers only if pretrained backbone is used
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)
