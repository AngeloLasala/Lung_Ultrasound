"""
Loss function for multi-class segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────── #
#  Individual loss functions (binary, kept for reference / other use cases)   #
# ─────────────────────────────────────────────────────────────────────────── #

def bce_loss(logits, targets, pos_weight=1.0):
    pw = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=pw)


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    targets = targets.float()
    bce     = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs   = torch.sigmoid(logits)
    p_t     = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * (1.0 - p_t) ** gamma * bce).mean()


# ─────────────────────────────────────────────────────────────────────────── #
#  Multiclass Dice loss                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def multiclass_dice_loss(logits, targets, smooth=1.0, ignore_index=-1):
    """
    Soft multiclass Dice Loss averaged over all classes (and the batch).

        DiceLoss_c = 1 - (2 * sum(p_c * t_c) + smooth)
                         / (sum(p_c) + sum(t_c) + smooth)

    Args:
        logits       : (B, C, H, W)  raw model output (pre-softmax).
        targets      : (B, H, W)     integer class indices, same as CE.
        smooth       : Laplace smoothing to avoid division by zero.
        ignore_index : class index to exclude from the mean (-1 = disabled).
    """
    num_classes = logits.size(1)
    probs       = F.softmax(logits, dim=1)                           # (B, C, H, W)

    targets_one_hot = F.one_hot(targets.long(), num_classes)         # (B, H, W, C)
    targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()   # (B, C, H, W)

    probs_flat   = probs.view(probs.size(0), num_classes, -1)        # (B, C, N)
    targets_flat = targets_one_hot.view(targets_one_hot.size(0), num_classes, -1)

    intersection   = (probs_flat * targets_flat).sum(dim=2)          # (B, C)
    denominator    = probs_flat.sum(dim=2) + targets_flat.sum(dim=2) # (B, C)
    dice_per_class = (2.0 * intersection + smooth) / (denominator + smooth)
    dice_per_class = dice_per_class.mean(dim=0)                      # (C,)

    if ignore_index >= 0:
        mask = torch.ones(num_classes, dtype=torch.bool, device=logits.device)
        mask[ignore_index] = False
        dice_per_class = dice_per_class[mask]

    return 1.0 - dice_per_class.mean()


# ─────────────────────────────────────────────────────────────────────────── #
#  CombinedCEDiceLoss — drop-in for nn.CrossEntropyLoss in train.py           #
# ─────────────────────────────────────────────────────────────────────────── #

class CombinedCEDiceLoss(nn.Module):
    """
    Cross-Entropy + soft multiclass Dice Loss.

        L = w_ce * CE(logits, targets) + w_dice * Dice(logits, targets)

    Drop-in replacement for nn.CrossEntropyLoss — same call signature,
    returns a plain scalar so train.py needs zero changes.

    In train.py, replace:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    with:
        criterion = CombinedCEDiceLoss(class_weights=class_weights)

    Args:
        class_weights (Tensor|None) : per-class weights forwarded to CE.
        w_ce          (float)       : weight of the CE term.   Default 1.0
        w_dice        (float)       : weight of the Dice term. Default 1.0
        smooth        (float)       : Dice smoothing constant. Default 1.0
        ignore_index  (int)         : class to skip in Dice.  -1 = disabled.
    """

    def __init__(self, class_weights=None, w_ce=1.0, w_dice=1.0,
                 smooth=1.0, ignore_index=-1):
        super().__init__()
        self.w_ce         = w_ce
        self.w_dice       = w_dice
        self.smooth       = smooth
        self.ignore_index = ignore_index
        self.ce           = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits, targets):
        """
        Args:
            logits  : (B, C, H, W)  raw model output.
            targets : (B, H, W)     integer class indices (same as CE).
        Returns:
            Scalar loss (differentiable).
        """
        l_ce   = self.ce(logits, targets.long())
        l_dice = multiclass_dice_loss(logits, targets, self.smooth, self.ignore_index)
        return self.w_ce * l_ce + self.w_dice * l_dice


# ─────────────────────────────────────────────────────────────────────────── #
#  Legacy binary CombinedLoss (BCE + Dice + Focal)                            #
# ─────────────────────────────────────────────────────────────────────────── #

class CombinedLoss(nn.Module):
    """
    Binary BCE + Dice + Focal for single-class (foreground/background) tasks.
    Returns (total_loss, breakdown_dict) for detailed logging.
    """

    def __init__(self, pos_weight=1.0, alpha=0.25, gamma=2.0,
                 smooth=1.0, w_bce=1.0, w_dice=1.0, w_focal=1.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.alpha  = alpha
        self.gamma  = gamma
        self.smooth = smooth
        self.w_bce  = w_bce
        self.w_dice = w_dice
        self.w_focal = w_focal

    def forward(self, logits, targets):
        l_bce   = bce_loss(logits, targets, pos_weight=self.pos_weight)
        l_dice  = multiclass_dice_loss(logits, targets.long(), self.smooth)
        l_focal = focal_loss(logits, targets, alpha=self.alpha, gamma=self.gamma)
        total   = self.w_bce * l_bce + self.w_dice * l_dice + self.w_focal * l_focal
        return total, {"bce": l_bce.item(), "dice": l_dice.item(),
                       "focal": l_focal.item(), "total": total.item()}

if __name__ == "__main__":
    B, C, H, W = 2, 3, 256, 256
    logits  = torch.randn(B, C, H, W)
    targets = torch.randint(0, C, (B, H, W))
    weights = torch.tensor([0.5, 1.5, 2.0])

    criterion = CombinedCEDiceLoss(class_weights=weights, w_ce=1.0, w_dice=1.0)
    loss = criterion(logits, targets)
    print(f"CE+Dice loss : {loss.item():.4f}")
    loss.backward()
    print("Backward pass OK")