import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from openpoints.utils import registry

LOSS = registry.Registry('loss')
LOSS.register_module(name='CrossEntropy', module=CrossEntropyLoss)
LOSS.register_module(name='CrossEntropyLoss', module=CrossEntropyLoss)

@LOSS.register_module()
class SmoothCrossEntropy(torch.nn.Module):
    def __init__(self, label_smoothing=0.2, 
                 ignore_index=None, 
                 num_classes=None, 
                 weight=None, 
                 return_valid=False
                 ):
        super(SmoothCrossEntropy, self).__init__()
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.return_valid = return_valid
        # Reduce label values in the range of logit shape
        if ignore_index is not None:
            reducing_list = torch.range(0, num_classes).long().cuda(non_blocking=True)
            inserted_value = torch.zeros((1, )).long().cuda(non_blocking=True)
            self.reducing_list = torch.cat([
                reducing_list[:ignore_index], inserted_value,
                reducing_list[ignore_index:]
            ], 0)
        if weight is not None:
            self.weight = torch.from_numpy(weight).float().cuda(
                non_blocking=True).squeeze()
        else:
            self.weight = None
            
    def forward(self, pred, gt):
        if len(pred.shape)>2:
            pred = pred.transpose(1, 2).reshape(-1, pred.shape[1])
        gt = gt.contiguous().view(-1)
        
        if self.ignore_index is not None: 
            valid_idx = gt != self.ignore_index
            pred = pred[valid_idx, :]
            gt = gt[valid_idx]        
            gt = torch.gather(self.reducing_list, 0, gt)
            
        if self.label_smoothing > 0:
            n_class = pred.size(1)
            one_hot = torch.zeros_like(pred).scatter(1, gt.view(-1, 1), 1)
            one_hot = one_hot * (1 - self.label_smoothing) + (1 - one_hot) * self.label_smoothing / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)
            if self.weight is not None:
                loss = -(one_hot * log_prb * self.weight).sum(dim=1).mean()
            else:
                loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gt, weight=self.weight)
        
        if self.return_valid:
            return loss, pred, gt
        else:
            return loss


@LOSS.register_module()
class MaskedCrossEntropy(torch.nn.Module):
    def __init__(self, label_smoothing=0.2):
        super(MaskedCrossEntropy, self).__init__()
        self.creterion = CrossEntropyLoss(label_smoothing=label_smoothing)
        
    def forward(self, logit, target, mask):
        logit = logit.transpose(1, 2).reshape(-1, logit.shape[1])
        target = target.flatten()
        mask = mask.flatten()
        idx = mask == 1
        loss = self.creterion(logit[idx], target[idx])
        return loss


def build_criterion_from_cfg(cfg, **kwargs):
    """
    Build a criterion (loss function), defined by cfg.NAME.
    Args:
        cfg (eDICT): 
    Returns:
        criterion: a constructed loss function specified by cfg.NAME
    """
    return LOSS.build(cfg, **kwargs)