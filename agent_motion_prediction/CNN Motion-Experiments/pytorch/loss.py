import torch 
from torch import nn

class pytorch_neg_multi_log_likelihood_batch(nn.Module): 
    """
        Compute a negative log-likelihood for the multi-modal scenario.
    """
    def __init__(self):
        super(pytorch_neg_multi_log_likelihood_batch, self).__init__()
    

    def forward(self, y, y_pred, avails): 
        """
        Args:
            y (Tensor): array of shape (bs)x(time)x(2D coords)
            y_pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
            confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
            avails (Tensor): array of shape (bs)x(time) with the availability for each y timestep
        Returns:
            Tensor: negative log-likelihood for this example, a single float number
        """

        # convert to (batch_size, num_modes, future_len, num_coords)
        y = torch.unsqueeze(y, 1)  # add modes
        avails = avails[:, None, :, None]  # add modes and cords

        # error (batch_size, num_modes, future_len)
        error = torch.sum(
            ((y - y_pred) * avails) ** 2, dim=-1
        )  # reduce coords and use availability


        # error (batch_size, num_modes)
        error = -torch.logsumexp(error, dim=-1, keepdim=True)

        return torch.mean(error)
    

def mean_displacement_error(y, y_pred, avails): 
    """
        Compute the mean displacement error between the ground truth and the prediction.
    """
    error = torch.sum(((y - y_pred) * avails) ** 2, dim=-1)  # reduce coords and use availability
    error = torch.sqrt(error)
    return torch.mean(error)

def final_displacement_error(y, y_pred, avails): 
    """
        Compute the final displacement error between the ground truth and the prediction.
    """
    error = torch.sum(((y[-1] - y_pred[-1]) * avails[-1]) ** 2, dim=-1)  # reduce coords and use availability
    error = torch.sqrt(error)
    return torch.mean(error)
