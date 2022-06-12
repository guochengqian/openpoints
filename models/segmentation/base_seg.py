"""
Author: PointNeXt
"""
import copy
import torch
import torch.nn as nn
import logging
from ...utils import get_missing_parameters_message, get_unexpected_parameters_message
from ..build import MODELS, build_model_from_cfg
from ..layers import create_linearblock, create_convblock1d


@MODELS.register_module()
class BaseSeg(nn.Module):
    def __init__(self,
                 encoder_args=None,
                 decoder_args=None,
                 cls_args=None,
                 **kwargs):
        super().__init__()
        self.encoder = build_model_from_cfg(encoder_args)
        if decoder_args is not None:
            decoder_args_merged_with_encoder = copy.deepcopy(encoder_args)
            decoder_args_merged_with_encoder.update(decoder_args)
            decoder_args_merged_with_encoder.encoder_channel_list = self.encoder.channel_list if hasattr(self.encoder, 'channel_list') else None
            self.decoder = build_model_from_cfg(decoder_args_merged_with_encoder) 
        else:
            self.decoder = None

        if cls_args is not None:
            if hasattr(self.decoder, 'out_channels'):
                in_channels = self.decoder.out_channels
            elif hasattr(self.encoder, 'out_channels'):
                in_channels = self.encoder.out_channels
            else:
                in_channels = cls_args.get('in_channels', None)
            cls_args.in_channels = in_channels
            self.head = build_model_from_cfg(cls_args)
        else:
            self.head = None

    def forward(self, p0, f0=None):
        if hasattr(p0, 'keys'):
            p0, f0 = p0['pos'], p0['x']
        else:
            if f0 is None:
                f0 = p0.transpose(1, 2).contiguous()
        p, f = self.encoder.forward_all_features(p0, f0)
        if self.decoder is not None:
            f = self.decoder(p, f).squeeze(-1)
        if self.head is not None:
            f = self.head(f)
        return f


    def get_loss(self, ret, gt):
        return self.criterion(ret, gt.long())

    def load_model_from_ckpt(self, ckpt_path, only_encoder=False):
        ckpt = torch.load(ckpt_path)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['model'].items()}

        if only_encoder:
            base_ckpt = {k.replace("encoder.", ""): v for k, v in base_ckpt.items()}
            incompatible = self.encoder.load_state_dict(base_ckpt, strict=False)
        else:
            incompatible = self.load_state_dict(base_ckpt, strict=False)
        if incompatible.missing_keys:
            logging.info('missing_keys')
            logging.info(
                get_missing_parameters_message(incompatible.missing_keys),
            )
        if incompatible.unexpected_keys:
            logging.info('unexpected_keys')
            logging.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys),

            )
        logging.info(f'Successful Loading the ckpt from {ckpt_path}')

# Shapenet Part Dataset is segmented with the prior of shape categories
@MODELS.register_module()
class BasePartSeg(nn.Module):
    def __init__(self,
                 encoder_args=None,
                 decoder_args=None,
                 cls_args=None,
                 smoothing=False, ignore_index=-100,
                 **kwargs):
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
        self.encoder = build_model_from_cfg(encoder_args)
        decoder_args_merged_with_encoder = copy.deepcopy(encoder_args)
        decoder_args_merged_with_encoder.update(decoder_args)
        self.decoder = build_model_from_cfg(decoder_args_merged_with_encoder)
        in_channels = self.decoder.out_channels if hasattr(self.decoder, 'out_channels') else cls_args.get('in_channels', None)
        cls_args.in_channels = in_channels
        assert cls_args.in_channels is not None
        self.head = build_model_from_cfg(cls_args)

    def forward(self, p0, f0=None):
        if hasattr(p0, 'keys'):
            p0, f0, cls0 = p0['pos'], p0['x'], p0['cls']
        p, f = self.encoder.forward_all_features(p0, f0)
        f = self.decoder(p, f, cls0).squeeze(-1)
        return self.head(f)

    def get_loss(self, ret, gt):
        return self.criterion(ret, gt.long())

    def load_model_from_ckpt(self, ckpt_path, only_encoder=False):
        ckpt = torch.load(ckpt_path)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['model'].items()}

        if only_encoder:
            base_ckpt = {k.replace("encoder.", ""): v for k, v in base_ckpt.items()}
            incompatible = self.encoder.load_state_dict(base_ckpt, strict=False)
        else:
            incompatible = self.load_state_dict(base_ckpt, strict=False)
        if incompatible.missing_keys:
            logging.info('missing_keys')
            logging.info(
                get_missing_parameters_message(incompatible.missing_keys),
            )
        if incompatible.unexpected_keys:
            logging.info('unexpected_keys')
            logging.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys),

            )
        logging.info(f'Successful Loading the ckpt from {ckpt_path}')



@MODELS.register_module()
class VariableSeg(BaseSeg):
    def __init__(self,
                 encoder_args=None,
                 decoder_args=None,
                 cls_args=None,
                 **kwargs):
        super().__init__(encoder_args, decoder_args, cls_args)
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")

    def forward(self, data):
        p, f, b = self.encoder.forward_all_features(data)
        f = self.decoder(p, f, b).squeeze(-1)
        return self.head(f)



@MODELS.register_module()
class SegHead(nn.Module):
    def __init__(self,
                 num_classes, in_channels,
                 mlps = None, 
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 dropout=0.5,
                 **kwargs
                 ):
        """A scene segmentation head for ResNet backbone.
        Args:
            num_classes: class num.
            in_channles: the base channel num.
        Returns:
            logits: (B, num_classes, N)
        """
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
        if mlps is None:
            mlps = [in_channels, in_channels] + [num_classes]
        else:
            mlps = [in_channels] + mlps + [num_classes]
        heads = []
        print(mlps, norm_args, act_args)
        for i in range(len(mlps) - 2):
            heads.append(create_convblock1d(mlps[i], mlps[i + 1],
                                            norm_args=norm_args,
                                            act_args=act_args))
            if dropout:
                heads.append(nn.Dropout(dropout))

        heads.append(create_convblock1d(mlps[-2], mlps[-1], act_args=None))
        self.head = nn.Sequential(*heads)

    def forward(self, end_points):
        logits = self.head(end_points)
        return logits
    

@MODELS.register_module()
class VariableSegHead(nn.Module):
    def __init__(self,
                 num_classes, in_channels,
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 dropout=0.5,
                 **kwargs
                 ):
        """A scene segmentation head for ResNet backbone.
        Args:
            num_classes: class num.
            in_channles: the base channel num.
        Returns:
            logits: (B, num_classes, N)
        """
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
        mlps = [in_channels, in_channels] + [num_classes]

        heads = []
        print(mlps, norm_args, act_args)
        for i in range(len(mlps) - 2):
            heads.append(create_linearblock(mlps[i], mlps[i + 1],
                                            norm_args=norm_args,
                                            act_args=act_args))
            if dropout:
                heads.append(nn.Dropout(dropout))

        heads.append(create_linearblock(mlps[-2], mlps[-1], act_args=None))
        self.head = nn.Sequential(*heads)

    def forward(self, end_points):
        logits = self.head(end_points)
        return logits