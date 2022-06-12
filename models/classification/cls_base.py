import torch
import torch.nn as nn
import logging
from typing import List
from openpoints.models.layers.norm import create_norm1d
from ..layers import create_linearblock
from ...utils import get_missing_parameters_message, get_unexpected_parameters_message
from ..build import MODELS, build_model_from_cfg


@MODELS.register_module()
class BaseCls(nn.Module):
    def __init__(self,
                 encoder_args=None,
                 cls_args=None,
                 **kwargs):
        super().__init__()
        self.encoder = build_model_from_cfg(encoder_args)
        in_channels = self.encoder.out_channels if hasattr(self.encoder, 'out_channels') else cls_args.get('in_channels', None)
        cls_args.in_channels = in_channels
        self.prediction = build_model_from_cfg(cls_args)

    def forward(self, p0, f0=None):
        global_feat = self.encoder.forward_cls_feat(p0, f0)
        return self.prediction(global_feat)

    def get_loss_acc(self, ret, gt):
        loss = self.criterion(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, ckpt_path, only_encoder=False):
        ckpt = torch.load(ckpt_path)
        base_ckpt=ckpt
        if  'model' in ckpt:
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
class ClsHead(nn.Module):
    def __init__(self,
                 num_classes: int, 
                 in_channels: int, 
                 mlps: List[int]=[256],
                 norm_args: dict=None,
                 act_args: dict={'act': 'relu'},
                 dropout: float=0.5,
                 cls_feat: str=None, 
                 **kwargs
                 ):
        """A general classification head. supports global pooling and [CLS] token
        Args:
            num_classes (int): class num
            in_channels (int): input channels size
            mlps (List[int], optional): channel sizes for hidden layers. Defaults to [256].
            norm_args (dict, optional): dict of configuration for normalization. Defaults to None.
            act_args (_type_, optional): dict of configuration for activation. Defaults to {'act': 'relu'}.
            dropout (float, optional): use dropout when larger than 0. Defaults to 0.5.
            cls_feat (str, optional): preprocessing input features to obtain global feature. 
                                      $\eg$ cls_feat='max,avg' means use the concatenateion of maxpooled and avgpooled features. 
                                      Defaults to None, which means the input feautre is the global feature
        Returns:
            logits: (B, num_classes, N)
        """
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
        self.cls_feat = cls_feat.split(',') if cls_feat is not None else None
        in_channels = len(self.cls_feat) * in_channels if cls_feat is not None else in_channels
        mlps = [in_channels] + mlps + [num_classes]

        heads = []
        for i in range(len(mlps) - 2):
            heads.append(create_linearblock(mlps[i], mlps[i + 1],
                                            norm_args=norm_args,
                                            act_args=act_args))
            if dropout:
                heads.append(nn.Dropout(dropout))
        heads.append(create_linearblock(mlps[-2], mlps[-1], act_args=None))
        self.head = nn.Sequential(*heads)


    def forward(self, end_points):
        if self.cls_feat is not None: 
            global_feats = [] 
            for preprocess in self.cls_feat:
                if 'max' in preprocess:
                    global_feats.append(torch.max(end_points, dim=1, keepdim=False)[0])
                elif preprocess in ['avg', 'mean']:
                    global_feats.append(torch.mean(end_points, dim=1, keepdim=False))
            end_points = torch.cat(global_feats, dim=1)
        logits = self.head(end_points)
        return logits