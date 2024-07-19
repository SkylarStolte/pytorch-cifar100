import warnings
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


from monai.networks import one_hot
from monai.utils import LossReduction
from monai.losses import DiceLoss


__all__ = [
    "hard_binned_calibration",
    "HardL1ACELoss",
    "HardL1ACEandCELoss",
    "HardL1ACEandDiceLoss",
    "HardL1ACEandDiceCELoss",
]


def hard_binned_calibration(input, target, num_bins=20, right=False):
    batch_size, num_channels = input.shape[:2]
    boundaries = torch.linspace(
        start=0.0,
        end=1.0 + torch.finfo(torch.float32).eps,
        steps=num_bins + 1,
        device=input.device,
    )

    mean_p_per_bin = torch.zeros(
        batch_size, num_channels, num_bins, device=input.device
    )
    mean_gt_per_bin = torch.zeros_like(mean_p_per_bin)
    bin_counts = torch.zeros_like(mean_p_per_bin)

    input = input.float()  # Ensure input and target are floats for calculations
    target = target.float()
    
    m = nn.Softmax(dim=1)
    input = m(input)

    # Clamp the input values to ensure they fall within the boundaries
    input_clamped = input.clamp(min=boundaries[0], max=boundaries[-1])

    # Vectorized binning
    bin_idx = torch.bucketize(input_clamped, boundaries[1:], right=right)

    # Flatten the tensors to perform binning and counting in a vectorized manner
    input_flat = input_clamped.view(batch_size * num_channels, -1)
    target_flat = target.view(batch_size * num_channels, -1)
    bin_idx_flat = bin_idx.view(batch_size * num_channels, -1)

    for i in range(num_bins):
        bin_mask = bin_idx_flat == i
        if bin_mask.any():
            count = bin_mask.sum(dim=-1)
            bin_counts.view(-1, num_bins)[:, i] = count
            mean_p_per_bin.view(-1, num_bins)[:, i] = (input_flat * bin_mask).sum(dim=-1) / (count + 1e-8)
            mean_gt_per_bin.view(-1, num_bins)[:, i] = (target_flat * bin_mask).sum(dim=-1) / (count + 1e-8)

    return mean_p_per_bin, mean_gt_per_bin, bin_counts


class HardL1ACELoss(_Loss):
    """
    Hard Binned L1 Average Calibration Error (ACE) loss for classification.
    """

    def __init__(
        self,
        num_bins: int = 20,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        reduction: LossReduction | str = LossReduction.MEAN,
        weight: Sequence[float] | float | int | torch.Tensor | None = None,
        right: bool = False,
    ) -> None:
        """
        Args:
            num_bins: the number of bins to use for the binned L1 ACE loss calculation. Defaults to 20.
            include_background: if False, class index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: callable function to execute other activation layers. Defaults to ``None``.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
            weight: weights to apply to the classes. If None, no weights are applied.
                The input can be a single value (same weight for all classes), a sequence of values (length
                equal to the number of classes), or a tensor. Defaults to None.
            right: If False (default), the bins include the left boundary and exclude the right boundary.

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(
                f"other_act must be None or callable but is {type(other_act).__name__}."
            )
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError(
                "Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None]."
            )
        self.num_bins = num_bins
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.weight = weight
        self.right = right
        self.register_buffer("class_weight", torch.ones(1))

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Tensor with shape [batch_size, num_classes, ...], representing predicted logits or probabilities.
            target: Tensor with the same shape as input, representing one-hot encoded ground truth labels.

        Raises:
            AssertionError: When input and target (after one hot transform if set)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        """
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_classes = input.shape[1]
        if self.softmax:
            if n_classes == 1:
                warnings.warn("Single class prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_classes == 1:
                warnings.warn("Single class prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_classes)

        if not self.include_background:
            if n_classes == 1:
                warnings.warn(
                    "Single class prediction, `include_background=False` ignored."
                )
            else:
                # Exclude background class
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(
                f"Ground truth has different shape ({target.shape}) from input ({input.shape})."
            )

        # Calculate ACE loss:
        mean_p_per_bin, mean_gt_per_bin, bin_counts = hard_binned_calibration(
            input, target, num_bins=self.num_bins, right=self.right
        )
        f = torch.nanmean(torch.abs(mean_p_per_bin - mean_gt_per_bin), dim=-1)

        if self.weight is not None and n_classes != 1:
            if isinstance(self.weight, (float, int)):
                self.class_weight = torch.full((n_classes,), self.weight)
            else:
                self.class_weight = torch.as_tensor(self.weight, device=input.device)
                if self.class_weight.shape[0] != n_classes:
                    raise ValueError(
                        "Length of `weight` should be equal to the number of classes."
                    )

            if torch.any(self.class_weight < 0):
                raise ValueError("Values of `weight` should be non-negative.")

            f = f * self.class_weight.to(f)

        if self.reduction == "mean":
            return torch.mean(f)

        if self.reduction == "sum":
            return torch.sum(f)

        return f


class HardL1ACEandCELoss(_Loss):
    """
    A class that combines L1 ACE Loss and CrossEntropyLoss with specified weights.
    """

    def __init__(
        self,
        ace_weight=0.5,
        ce_weight=0.5,
        to_onehot_y=False,
        ace_params=None,
        ce_params=None,
    ):
        """
        Initializes the HardL1ACEandCELoss class.

        Args:
            ace_weight (float): Weight for the HardL1ACELoss component.
            ce_weight (float): Weight for the CrossEntropyLoss component.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `pred` (``pred.shape[1]``). Defaults to False.
            ace_params (dict, optional): Parameters for the HardL1ACELoss.
            ce_params (dict, optional): Parameters for the CrossEntropyLoss.
        """
        super().__init__()
        self.ace_weight = ace_weight
        self.ce_weight = ce_weight
        self.to_onehot_y = to_onehot_y
        self.ace_loss = HardL1ACELoss(**(ace_params if ace_params is not None else {}))
        self.ce_loss = nn.CrossEntropyLoss(
            **(ce_params if ce_params is not None else {})
        )

    def forward(self, y_pred, y_true):
        """
        Forward pass for calculating the weighted sum of L1 ACE and CrossEntropy losses.

        Args:
            y_pred: Predicted logits or probabilities.
            y_true: Ground truth labels.

        Returns:
            The weighted sum of L1 ACE and CrossEntropy losses.
        """
        # TODO: need to think about how reductions are handles for the two losses when combining
        if self.to_onehot_y:
            y_true = one_hot(y_true, num_classes=y_pred.shape[1])
        ace_loss_val = self.ace_loss(y_pred, y_true)
        ce_loss_val = self.ce_loss(y_pred, y_true)
        return self.ace_weight * ace_loss_val + self.ce_weight * ce_loss_val


class HardL1ACEandDiceLoss(_Loss):
    """
    A class that combines L1 ACE Loss and DiceLoss with specified weights.
    """

    def __init__(
        self,
        ace_weight=0.5,
        dice_weight=0.5,
        to_onehot_y=False,
        ace_params=None,
        dice_params=None,
    ):
        """
        Initializes the HardL1ACEandCELoss class.

        Args:
            ace_weight (float): Weight for the HardL1ACELoss component.
            dice_weight (float): Weight for the DiceLoss component.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `pred` (``pred.shape[1]``). Defaults to False.
            ace_params (dict, optional): Parameters for the HardL1ACELoss.
            dice_params (dict, optional): Parameters for the DiceLoss.
        """
        super().__init__()
        self.ace_weight = ace_weight
        self.dice_weight = dice_weight
        self.to_onehot_y = to_onehot_y
        self.ace_loss = HardL1ACELoss(**(ace_params if ace_params is not None else {}))
        self.dice_loss = DiceLoss(**(dice_params if dice_params is not None else {}))

    def forward(self, y_pred, y_true):
        """
        Forward pass for calculating the weighted sum of L1 ACE and Dice losses.

        Args:
            y_pred: Predicted logits or probabilities.
            y_true: Ground truth labels.

        Returns:
            The weighted sum of L1 ACE and Dice losses.
        """
        if self.to_onehot_y:
            y_true = one_hot(y_true, num_classes=y_pred.shape[1])
        ace_loss_val = self.ace_loss(y_pred, y_true)
        dice_loss_val = self.dice_loss(y_pred, y_true)
        return self.ace_weight * ace_loss_val + self.dice_weight * dice_loss_val


class HardL1ACEandDiceCELoss(_Loss):
    """
    A class that combines L1 ACE Loss, Dice Loss, and CrossEntropyLoss with specified weights.
    """

    def __init__(
        self,
        ace_weight=0.33,
        ce_weight=0.33,
        dice_weight=0.33,
        to_onehot_y=False,
        ace_params=None,
        dice_params=None,
        ce_params=None,
    ):
        """
        Initializes the HardL1ACEandDiceCELoss class.

        Args:
            ace_weight (float): Weight for the HardL1ACELoss component.
            dice_weight (float): Weight for the DiceLoss component.
            ce_weight (float): Weight for the CrossEntropyLoss component.
            to_onehot_y (bool): Whether to convert the `target` into the one-hot format.
            ace_params (dict, optional): Parameters for the HardL1ACELoss.
            dice_params (dict, optional): Parameters for the DiceLoss.
            ce_params (dict, optional): Parameters for the CrossEntropyLoss.
        """
        super().__init__()
        self.ace_weight = ace_weight
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.to_onehot_y = to_onehot_y

        self.ace_loss = HardL1ACELoss(**(ace_params if ace_params is not None else {}))
        self.dice_loss = DiceLoss(**(dice_params if dice_params is not None else {}))
        self.ce_loss = nn.CrossEntropyLoss(
            **(ce_params if ce_params is not None else {})
        )

    def forward(self, y_pred, y_true):
        """
        Forward pass for calculating the weighted sum of L1 ACE, Dice, and CrossEntropy losses.

        Args:
            y_pred: Predicted logits or probabilities.
            y_true: Ground truth labels.

        Returns:
            The weighted sum of L1 ACE, Dice, and CrossEntropy losses.
        """
        if self.to_onehot_y:
            y_true = one_hot(y_true, num_classes=y_pred.shape[1])
        ace_loss_val = self.ace_loss(y_pred, y_true)
        dice_loss_val = self.dice_loss(y_pred, y_true)
        ce_loss_val = self.ce_loss(y_pred, y_true)
        return (
            self.ace_weight * ace_loss_val
            + self.dice_weight * dice_loss_val
            + self.ce_weight * ce_loss_val
        )
