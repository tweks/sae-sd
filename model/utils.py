import math
import torch.utils
from tqdm import tqdm
import torch

from pipeline.utils import set_logger, get_device

_, logger = set_logger(level="INFO")


class LinearDecayLR(torch.optim.lr_scheduler.LambdaLR):
    """
    Learning rate scheduler with linear decay after a constant period.
    
    Keeps the learning rate constant for a portion of training,
    then linearly decays to zero.
    
    Args:
        optimizer: The optimizer to modify
        total_epochs: Total number of training epochs
        decay_time: Fraction of training after which to start decay (0-1)
        last_epoch: The index of the last epoch (-1 for initialization)
    """
    def __init__(
        self, 
        optimizer: torch.optim.Optimizer, 
        total_epochs: int, 
        decay_time: float = 0.8, 
        last_epoch: int = -1
    ):
        def lr_lambda(epoch: int) -> float:
            """Calculate the learning rate multiplier for the current epoch."""
            if epoch < int(decay_time * total_epochs):
                return 1.0
            return max(0.0, (total_epochs - epoch) / ((1-decay_time) * total_epochs))

        super().__init__(optimizer, lr_lambda, last_epoch)


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Custom scheduler that implements:
    1. Linear warmup from initial_lr (x*0.1) to max_lr (x) in the first epoch
    2. Cosine annealing from max_lr (x) to final_lr (x*0.1) for remaining epochs
    
    Args:
        optimizer: The optimizer to modify
        max_lr: The maximum learning rate (x)
        total_epochs: Total number of epochs for training
        warmup_epoch: Number of epochs for warmup (default: 1)
        final_lr_factor: Factor for final learning rate (default: 0.1)
        last_epoch: The index of last epoch (default: -1)
    """
    def __init__(self, optimizer, max_lr, total_epochs, warmup_epoch=1, 
                 final_lr_factor=0.1, last_epoch=-1):
        self.max_lr = max_lr
        self.total_epochs = total_epochs
        self.warmup_epoch = warmup_epoch
        self.initial_lr = max_lr * final_lr_factor
        self.final_lr = max_lr * final_lr_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            logger.warning("To get the last learning rate computed by the scheduler, "
                        "please use `get_last_lr()`.")

        # During warmup (first epoch)
        if self.last_epoch < self.warmup_epoch:
            # Linear interpolation from initial_lr to max_lr
            alpha = self.last_epoch / self.warmup_epoch
            return [self.initial_lr + (self.max_lr - self.initial_lr) * alpha 
                    for _ in self.base_lrs]
        
        # After warmup - Cosine annealing
        else:
            # Adjust epoch count to start cosine annealing after warmup
            current = self.last_epoch - self.warmup_epoch
            total = self.total_epochs - self.warmup_epoch
            
            # Implement cosine annealing
            cosine_factor = (1 + math.cos(math.pi * current / total)) / 2
            return [self.final_lr + (self.max_lr - self.final_lr) * cosine_factor 
                    for _ in self.base_lrs]


def normalize_data(
    x: torch.Tensor, 
    eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize data by subtracting mean and dividing by standard deviation.
    
    Args:
        x: Input tensor
        eps: Small value to prevent division by zero
        
    Returns:
        Tuple of (normalized_data, mean, std)
    """
    mu = x.mean(dim=-1, keepdim=True)
    x_centered = x - mu
    std = x_centered.std(dim=-1, keepdim=True)
    x_normalized = x_centered / (std + eps)
    return x_normalized, mu, std


@torch.no_grad()
def geometric_median(
    dataset: torch.utils.data.Dataset,
    eps: float = 1e-5,
    device: torch.device | None = None,
    max_samples: int = 1000000, 
    max_iter: int = 100,
    do_first: bool = True,
) -> torch.Tensor:
    # Get device
    if device is None:
        device = get_device()
    
    indices = torch.randperm(len(dataset))[:min(len(dataset), max_samples)]
    if isinstance(dataset[0], (list, tuple)):
        id = 0 if do_first else 1
        points = torch.stack([dataset[i.item()][id] for i in indices], dim=0)
    else:
        points = torch.stack([dataset[i.item()] for i in indices], dim=0)
    points = points.to(device)
    
    y = torch.mean(points, dim=0)
    progress_bar = tqdm(range(max_iter), desc="Geometric Median Iteration", leave=False)

    # Weiszfeld algorithm
    for _ in progress_bar:
        # Calculate distances from current estimate
        D = torch.norm(points - y, dim=1)
        
        # Identify non-zero distances
        nonzeros = (D > eps)
        
        # If all points are very close to current estimate, we're done
        if not nonzeros.any():
            return y
        
        # Calculate weights for non-zero distances
        Dinv = 1 / D[nonzeros]
        Dinv_sum = torch.sum(Dinv)
        W = Dinv / Dinv_sum
        
        # Calculate weighted centroid
        T = torch.sum(W.view(-1, 1) * points[nonzeros], dim=0)
        
        # Handle case where some points coincide with current estimate
        num_zeros = len(points) - torch.sum(nonzeros)
        if num_zeros == 0:
            y_new = T
        else:
            # Adjust for coincident points
            R = (T * Dinv_sum) / (Dinv_sum + num_zeros)
            
            # Check for convergence
            r = torch.norm(y - R)
            progress_bar.set_postfix({"r": r.item()})
            
            if r < eps:
                return R # y
                
            y_new = R
        
        # Check for convergence
        if torch.norm(y - y_new) < eps:
            return y_new
            
        y = y_new
    
    # Return best estimate if max iterations reached
    return y


def calculate_grad_norm(model: torch.nn.Module) -> torch.Tensor:
    """
    Calculate the L2 norm of all gradients in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Gradient norm
    """
    # Initialize norm tensor on the right device
    total_norm = torch.tensor(0.0)
    
    # Get parameters with gradients
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    
    # Sum squared norms
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm ** 2
        
    # Take square root
    total_norm = total_norm ** 0.5
    return total_norm


def identify_dead_neurons(
    latent_bias: torch.Tensor, 
    threshold: float = 1e-5
) -> torch.Tensor:
    """
    Identify neurons that are effectively "dead" based on low activation bias.
    
    Args:
        latent_bias: Bias tensor for latent layer
        threshold: Bias magnitude threshold below which a neuron is considered dead
        
    Returns:
        Tensor of indices for dead neurons
    """
    dead_neurons = torch.where(torch.abs(latent_bias) < threshold)[0]
    return dead_neurons


def calculate_similarity_metrics(
    original_matrix: torch.Tensor, 
    reconstruction_matrix: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate similarity metrics between original and reconstructed data.
    
    Args:
        original_matrix: Original data
        reconstruction_matrix: Reconstructed data
        
    Returns:
        Tuple of (mean_cosine_similarity, mean_euclidean_distance)
    """
    # Calculate cosine similarity for each pair
    # First normalize the vectors
    original_norm = original_matrix.norm(dim=-1, keepdim=True)
    reconstruction_norm = reconstruction_matrix.norm(dim=-1, keepdim=True)

    # Avoid division by zero
    original_normalized = original_matrix / original_norm
    reconstruction_normalized = reconstruction_matrix / reconstruction_norm

    # Calculate dot product of normalized vectors
    cosine_similarities = torch.bmm(
        reconstruction_normalized.unsqueeze(1),
        original_normalized.unsqueeze(2)
    ).squeeze()

    # Calculate Euclidean distance for each pair
    euclidean_distances = torch.norm(original_matrix - reconstruction_matrix, dim=-1)

    return torch.mean(cosine_similarities), torch.mean(euclidean_distances)


def calculate_vector_mean(dataset: torch.utils.data.Dataset,
                          batch_size: int = 10000,
                          num_workers: int = 4,
                          do_first: bool = True) -> torch.Tensor:
    """
    Efficiently calculate the mean of vectors in a dataset.
    
    Args:
        dataset: PyTorch Dataset object
        batch_size: Batch size for iteration
        num_workers: Number of workers for data loading
        do_first: Whether to do the first element from the tuple of batch.
            If batch is not a tuple, this is ignored.
        
    Returns:
        Mean vector
    """
    # Use DataLoader to efficiently iterate through the dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False  # No need to shuffle for calculating mean
    )
    
    # Initialize sum and count
    if isinstance(dataset[0], (list, tuple)):
        # If dataset is a tuple, get the first element
        if do_first:
            vector_sum = torch.zeros_like(dataset[0][0])
        else:
            vector_sum = torch.zeros_like(dataset[0][1])
    else:
        vector_sum = torch.zeros_like(dataset[0])
    count = 0
    
    # Iterate through batches
    for batch in tqdm(dataloader, desc="Calculating Mean Vector", leave=False):
        if isinstance(batch, (list, tuple)):
            # If batch is a tuple, get the first element
            if do_first:
                batch = batch[0]
            else:
                batch = batch[1]

        # Get batch size from actual data
        batch_count = batch.size(0)
        
        # Sum vectors along batch dimension
        vector_sum += batch.sum(dim=0)
        count += batch_count
    
    # Calculate mean
    mean_vector = vector_sum / count
    
    return mean_vector


def rectangle(x: torch.Tensor) -> torch.Tensor:
    return ((x > -0.5) & (x < 0.5)).to(x)

class RectangleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x <= -0.5) | (x >= 0.5)] = 0
        return grad_input

class JumpReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        ctx.save_for_backward(x, log_threshold, torch.tensor(bandwidth))
        threshold = torch.exp(log_threshold)
        return x * (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, log_threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        threshold = torch.exp(log_threshold)
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth

class StepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        ctx.save_for_backward(x, log_threshold, torch.tensor(bandwidth))
        threshold = torch.exp(log_threshold)
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, log_threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        threshold = torch.exp(log_threshold)
        x_grad = torch.zeros_like(x)
        threshold_grad = (
            -(1.0 / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth