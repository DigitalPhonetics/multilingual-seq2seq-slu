import torch
import torch.nn.functional as F
from typing import Optional


class SquareAlignmentBaseLoss(torch.nn.Module):
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        out_masks: Optional[torch.Tensor],
        pred_lengths: Optional[torch.Tensor],
    ) -> torch.Tensor:
        max_length = max(target.size(1), pred.size(1))
        target = F.pad(target, (0, 0, 0, max_length - target.size(1)))
        pred = F.pad(pred, (0, 0, 0, max_length - pred.size(1)))

        if self.normalize:
            target = F.normalize(target, dim=2)
            pred = F.normalize(pred, dim=2)

        alignment_pred = torch.bmm(pred, target.permute(0, 2, 1))

        if self.eyed:
            alignment_target = torch.eye(
                target.size(1), device=target.device, dtype=target.dtype
            )
            alignment_target[alignment_target == 0] = -1
            alignment_target = alignment_target.unsqueeze(0).repeat(pred.size(0), 1, 1)
        else:
            alignment_target = torch.bmm(target, target.permute(0, 2, 1))

        if self.mask and out_masks is not None:
            if out_masks.size(1) < max_length:
                out_masks = F.pad(out_masks, (0, max_length - out_masks.size(1)))
            out_masks_ext = torch.mul(
                out_masks.unsqueeze(2), out_masks.unsqueeze(2).rot90(1, [1, 2])
            ).bool()
            alignment_pred = alignment_pred.masked_select(out_masks_ext)
            alignment_target = alignment_target.masked_select(out_masks_ext)

        return self.loss(alignment_pred, alignment_target)


class SquareAlignmentL1Loss(SquareAlignmentBaseLoss):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.L1Loss()
        self.normalize = False
        self.mask = False
        self.eyed = False


class SquareAlignmentL2Loss(SquareAlignmentBaseLoss):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss()
        self.normalize = False
        self.mask = False
        self.eyed = False


class NormalizedSquareAlignmentL1Loss(SquareAlignmentBaseLoss):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.L1Loss()
        self.normalize = True
        self.mask = False
        self.eyed = False


class NormalizedSquareAlignmentL2Loss(SquareAlignmentBaseLoss):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss()
        self.normalize = True
        self.mask = False
        self.eyed = False


class MaskedNormalizedSquareAlignmentL2Loss(SquareAlignmentBaseLoss):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss()
        self.normalize = True
        self.mask = True
        self.eyed = False


class EyedMaskedNormalizedSquareAlignmentL2Loss(SquareAlignmentBaseLoss):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss()
        self.normalize = True
        self.mask = True
        self.eyed = True
