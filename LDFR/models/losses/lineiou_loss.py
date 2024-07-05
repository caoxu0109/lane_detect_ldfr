import torch


def line_iou(pred, target, img_w, length=15, aligned=True):
    '''
    Calculate the line iou value between predictions and targets
    Args:
        pred: lane predictions, shape: (num_pred, 72)
        target: ground truth, shape: (num_target, 72)
        img_w: image width
        length: extended radius
        aligned: True for iou loss calculation, False for pair-wise ious in assign
    '''
    n_strips = pred.shape[1] - 1
    dy = 590 / n_strips * 2  # two horizontal grids
    _pred = pred.clone().detach()
    pred_dx = (
                      _pred[:, 2:] - _pred[:, :-2]
              ) * img_w  # pred x difference across two horizontal grids
    pred_width = length * torch.sqrt(pred_dx.pow(2) + dy ** 2) / dy
    pred_width = torch.cat(
        [pred_width[:, 0:1], pred_width, pred_width[:, -1:]], dim=1
    )
    target_dx = (target[:, 2:] - target[:, :-2]) * img_w
    target_dx[torch.abs(target_dx) > 1e4] = 0
    target_width = length * torch.sqrt(target_dx.pow(2) + dy ** 2) / dy
    target_width = torch.cat(
        [target_width[:, 0:1], target_width, target_width[:, -1:]], dim=1
    )
    px1 = pred - pred_width
    px2 = pred + pred_width
    tx1 = target - target_width
    tx2 = target + target_width
    if aligned:
        invalid_mask = target
        ovr = torch.min(px2, tx2) - torch.max(px1, tx1)
        union = torch.max(px2, tx2) - torch.min(px1, tx1)
        #G = torch.clamp((union - 4*length) / (union + 1e-9),min=0.,max=1.)
    else:
        num_pred = pred.shape[0]
        invalid_mask = target.repeat(num_pred, 1, 1)
        ovr = (torch.min(px2[:, None, :], tx2[None, ...]) -
               torch.max(px1[:, None, :], tx1[None, ...]))
        union = (torch.max(px2[:, None, :], tx2[None, ...]) -
                 torch.min(px1[:, None, :], tx1[None, ...]))
        G = torch.clamp((union - 4*length) / (union + 1e-9),min=0.,max=1.)
    invalid_masks = (invalid_mask < 0) | (invalid_mask >= img_w)
    ovr[invalid_masks] = 0.
    union[invalid_masks] = 0.
    # G[invalid_masks] = 0.
    iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9) #- G.sum(dim=-1)
    return iou


def liou_loss(pred, target, img_w, length=15):
    return (1 - line_iou(pred, target, img_w, length)).mean()