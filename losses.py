import torch
import torch.nn.functional as F


def gaussian_nll_loss(input, target, var, reduction="mean", eps: float = 1e-6):
    # Check validity of reduction mode
    if reduction != "none" and reduction != "mean" and reduction != "sum":
        raise ValueError(reduction + " is not valid")

    # Entries of var must be non-negative
    if torch.any(var < 0):
        raise ValueError("var has negative entry/entries")

    # Clamp for stability
    var = var.clone()
    with torch.no_grad():
        var.clamp_(min=eps)
    # Calculate the loss
    loss = 0.5 * (torch.log(var) + (input - target) ** 2 / var)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def asymetric_gaussian_nll_loss(
    input, target, var_left, var_right, reduction="mean", eps: float = 1e-6
):
    # Check validity of reduction mode
    if reduction != "none" and reduction != "mean" and reduction != "sum":
        raise ValueError(reduction + " is not valid")

    # Entries of var must be non-negative
    if torch.any(var_left < 0) or torch.any(var_right < 0):
        raise ValueError("var has negative entry/entries")

    # Clamp for stability
    var_left = var_left.clone()
    var_right = var_right.clone()
    with torch.no_grad():
        var_left.clamp_(min=eps)
        var_right.clamp_(min=eps)
    # Calculate the loss on each side of the average
    left_loss = F.relu(input - target) ** 2 / var_left
    right_loss = F.relu(target - input) ** 2 / var_right
    # Calculate the total loss
    loss = 0.5 * (torch.log(var_left + var_right) + left_loss + right_loss)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss
