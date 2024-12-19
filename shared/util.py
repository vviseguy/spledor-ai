
import torch
import torch.nn.functional as F

def pad_tensor_to_shape(tensor: torch.Tensor, target_shape: tuple, pad_side: str = 'end') -> torch.Tensor:
    """
    Pads the input tensor with zeros to match the target shape.

    Args:
        tensor (torch.Tensor): The input tensor to pad.
        target_shape (tuple): The desired shape after padding.
        pad_side (str): Where to apply padding. Options: 'begin', 'end', 'both'.
                        'begin' - pad at the beginning of each dimension.
                        'end' - pad at the end of each dimension.
                        'both' - pad equally on both sides (if possible).

    Returns:
        torch.Tensor: The padded tensor.

    Raises:
        ValueError: If target_shape is smaller than tensor.shape in any dimension.
    """
    if len(target_shape) != len(tensor.shape):
        raise ValueError("Target shape must have the same number of dimensions as the tensor.")
    
    # Calculate padding for each dimension
    pad = []
    for i in range(len(tensor.shape)-1, -1, -1):  # Start from the last dimension
        orig_dim = tensor.shape[i]
        target_dim = target_shape[i]
        if target_dim < orig_dim:
            raise ValueError(f"Target dimension {target_dim} is smaller than original dimension {orig_dim} at axis {i}.")
        pad_size = target_dim - orig_dim
        if pad_size == 0:
            pad.extend([0, 0])
        else:
            if pad_side == 'begin':
                pad.extend([pad_size, 0])
            elif pad_side == 'end':
                pad.extend([0, pad_size])
            elif pad_side == 'both':
                pad_before = pad_size // 2
                pad_after = pad_size - pad_before
                pad.extend([pad_before, pad_after])
            else:
                raise ValueError("pad_side must be 'begin', 'end', or 'both'.")
    
    # Apply padding
    padded_tensor = F.pad(tensor, pad, mode='constant', value=0)
    return padded_tensor
