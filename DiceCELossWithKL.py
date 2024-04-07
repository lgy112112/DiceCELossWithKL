# you need monai
# and torch
# and git clone https://github.com/by-liu/SegLossBias.git
# then import sys
# sys.path.append('/kaggle/working/SegLossBias')
# and so on

class DiceCELossWithKL(_Loss):
    """
    A compound loss function that combines Dice loss, Cross-Entropy loss, and KL divergence.
    This loss is designed for segmentation tasks with added regularization via KL divergence to 
    ensure the predicted probabilities match the ground truth distribution more closely.

    Attributes:
        mode (str): Mode for handling different classification scenarios.
        include_background (bool): If true, include background in loss calculation.
        to_onehot_y (bool): If true, converts target into one-hot encoding.
        sigmoid (bool): If true, apply sigmoid activation to the predictions.
        softmax (bool): If true, apply softmax activation to the predictions.
        other_act (Callable): Custom activation function to be applied.
        squared_pred (bool): If true, square the predictions in Dice loss.
        jaccard (bool): If true, use Jaccard index as the basis for Dice loss.
        reduction (str): Specifies the reduction to apply to the output: 'none', 'mean', 'sum'.
        smooth_nr (float): Smoothing factor numerator for Dice loss.
        smooth_dr (float): Smoothing factor denominator for Dice loss.
        batch (bool): If true, calculate losses per batch.
        ce_weight (torch.Tensor): Weights for Cross-Entropy loss.
        weight (torch.Tensor): General weights used across losses.
        lambda_dice (float): Weight factor for Dice loss.
        lambda_ce (float): Weight factor for Cross-Entropy loss.
        lambda_kl (float): Weight factor for KL divergence.
        temp (float): Temperature scaling factor for KL divergence.
    """
    def __init__(self, mode='MULTICLASS_MODE', include_background=True, to_onehot_y=False, sigmoid=False, softmax=False,
                 other_act=None, squared_pred=False, jaccard=False, reduction="mean", smooth_nr=1e-5, smooth_dr=1e-5,
                 batch=False, ce_weight=None, weight=None, lambda_dice=1.0, lambda_ce=1.0, lambda_kl=1.0, temp=10.0):
        super().__init__()
        self.mode = mode
        self.temp = temp
        reduction = look_up_option(reduction, DiceCEReduction).value
        weight = ce_weight if ce_weight is not None else weight
        dice_weight = None
        if weight is not None and not include_background:
            dice_weight = weight[1:]
        else:
            dice_weight = weight

        self.dice = DiceLoss(include_background=include_background, to_onehot_y=to_onehot_y, sigmoid=sigmoid,
                             softmax=softmax, other_act=other_act, squared_pred=squared_pred, jaccard=jaccard,
                             reduction=reduction, smooth_nr=smooth_nr, smooth_dr=smooth_dr, batch=batch, weight=dice_weight)
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
        self.binary_cross_entropy = nn.BCEWithLogitsLoss(pos_weight=weight, reduction=reduction)
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_ce < 0.0:
            raise ValueError("lambda_ce should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        self.old_pt_ver = not pytorch_after(1, 10)
        self.lambda_kl = lambda_kl

    def ce(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute CrossEntropy loss for the input logits and the target.
        Adjusts target format if necessary to fit PyTorch CrossEntropyLoss requirements.
        """
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch != n_target_ch and n_target_ch == 1:
            target = torch.squeeze(target, dim=1)
            target = target.long()
        elif self.old_pt_ver:
            warnings.warn(
                f"Multichannel targets are not supported in this older Pytorch version {torch.__version__}. "
                "Using argmax (as a workaround) to convert target to a single channel."
            )
            target = torch.argmax(target, dim=1)
        elif not torch.is_floating_point(target):
            target = target.to(dtype=input.dtype)

        return self.cross_entropy(input, target)  # type: ignore[no-any-return]

    def bce(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Binary CrossEntropy loss for the input logits and the target.
        Adjusts target format if necessary to fit PyTorch BCEWithLogitsLoss requirements.
        """
        if not torch.is_floating_point(target):
            target = target.to(dtype=input.dtype)

        return self.binary_cross_entropy(input, target)  # type: ignore[no-any-return]

    def kl_div(self, p, q):
        """
        Calculate KL divergence between two probability distributions.
        Ensures numerical stability by adding a small constant to the denominators.
        """
        kl = p * torch.log((p + 1e-10) / (q + 1e-10))
        return kl.sum()

    def convert_to_one_hot(self, targets, num_classes):
        """
        Convert target tensor into one-hot format.
        Adjusts dimensions to match model output dimensions.
        """
        targets = torch.squeeze(targets, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2)
        return targets_one_hot
    
    def calculate_gt_proportion(self, targets_one_hot):
        """
        Calculate the proportion of each class in the ground truth.
        """
        class_totals = targets_one_hot.sum(dim=[0, 2, 3])  # Sum across batches and spatial dimensions
        epsilon = 1e-8  # Small constant to avoid division by zero
        gt_proportion = class_totals / (class_totals.sum() + epsilon)
        return gt_proportion
    
    def calculate_pred_proportion(self, preds):
        """
        Calculate the proportion of each class in the predictions.
        """
        pred_totals = preds.sum(dim=[0, 2, 3])  # Sum across batches and spatial dimensions
        epsilon = 1e-8  # Small constant to avoid division by zero
        pred_proportion = pred_totals / (pred_totals.sum() + epsilon)
        return pred_proportion

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the compound loss function. Calculates and aggregates Dice loss, Cross-Entropy loss, 
        and KL divergence based on the defined weights.
        """
        if len(input.shape) != len(target.shape):
            raise ValueError("The number of dimensions for input and target should be the same.")
        
        dice_loss = self.dice(input, target)
        ce_loss = self.ce(input, target) if input.shape[1] != 1 else self.bce(input, target)
        preds = F.softmax(input, dim=1)
        target_one_hot = self.convert_to_one_hot(target, num_classes=input.size(1))
        gt_proportion = self.calculate_gt_proportion(target_one_hot)
        pred_proportion = self.calculate_pred_proportion(preds)
        kl_loss = self.kl_div(gt_proportion, pred_proportion)
        
        total_loss = self.lambda_dice * dice_loss + self.lambda_ce * ce_loss + self.lambda_kl * kl_loss
        return total_loss

