from typing import Callable, Any
from functools import partial
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import normalize_data, JumpReLUFunction, StepFunction


def elu1p(x, inplace: bool = False):
    return F.elu(x, 1.0, inplace) + 1.0

class ELU1p(nn.Module):
    def forward(self, x, inplace: bool = False):
        return elu1p(x, inplace)


class SoftCapping(nn.Module):
    def __init__(self, soft_cap):
        super(SoftCapping, self).__init__()
        self.soft_cap = soft_cap

    def forward(self, logits):
        return self.soft_cap * torch.tanh(logits / self.soft_cap)


class TopK(nn.Module):
    def __init__(self, k: int, act_fn: Callable = nn.Identity(), use_abs: bool = False) -> None:
        super().__init__()
        self.k = k
        self.act_fn = act_fn
        self.use_abs = use_abs
    
    def extra_repr(self) -> str:
        """Return string representation of module parameters."""
        return f"k={self.k}, act_fn={self.act_fn}, use_abs={self.use_abs}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use absolute values if requested
        values = torch.abs(x) if self.use_abs else x
        
        # Get indices of top-k elements
        _, indices = torch.topk(values, k=min(self.k, x.shape[-1]), dim=-1)
        
        # Gather original values at those indices
        top_values = torch.gather(x, -1, indices)
        
        # Apply activation function to those values
        activated_values = self.act_fn(top_values)
        
        # Create output tensor of zeros and place activated values at correct positions
        result = torch.zeros_like(x)
        result.scatter_(-1, indices, activated_values)
        
        # Verify that we have at most k non-zero elements per sample
        return result

    
    def forward_eval(self,x: torch.Tensor) -> torch.Tensor:
        x = torch.abs(x) if self.use_abs else x
        x = self.act_fn(x)
        return x


class BatchTopK(TopK):
    def __init__(self, k: int, act_fn: Callable = nn.Identity(), use_abs: bool = False) -> None:
        # Call the parent class constructor
        super().__init__(k, act_fn, use_abs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get batch size
        batch_size = x.shape[0]
        
        # Calculate total number of values to keep
        total_k = min(self.k * batch_size, x.numel())
        
        # Use absolute values if requested for selection
        values = torch.abs(x) if self.use_abs else x
        
        # Store original shape and flatten
        flat_values = values.flatten()
        flat_x = x.flatten()
        
        # Get indices of top-k elements across the entire batch
        _, indices = torch.topk(flat_values, k=total_k, dim=-1)
        
        # Create output tensor of zeros and place original values at correct positions
        flat_result = torch.zeros_like(flat_x)
        
        # Apply activation function to selected values and place them in the result
        activated_values = self.act_fn(flat_x[indices])
        flat_result.scatter_(-1, indices, activated_values)
        
        # Reshape back to original shape
        result = flat_result.reshape(values.shape)
        
        return result
    
    def forward_eval(self,x: torch.Tensor) -> torch.Tensor:
        x = torch.abs(x) if self.use_abs else x
        x = self.act_fn(x)
        return x


class JumpReLU(nn.Module):
    def __init__(self, hidden_dim: int, init_threshold: float=0.001, bandwidth: float=0.001) -> None:
        """
        Initialize JumpReLU activation with specified parameters.

        Args:
            hidden_dim: Dimension of the input tensor.
            init_threshold: Initial threshold for the JUMP mechanism.
        """
        super().__init__()
        self.log_threshold = nn.Parameter(torch.full((hidden_dim,), np.log(init_threshold)))
        self.bandwidth = bandwidth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_relu = torch.relu(x)
        return JumpReLUFunction.apply(x_relu, self.log_threshold, self.bandwidth)
    
    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        return StepFunction.apply(x, self.log_threshold, self.bandwidth)


ACTIVATIONS_CLASSES = {
    "ReLU": nn.ReLU,
    "ELU1p": ELU1p,
    "Identity": nn.Identity,
    "TopK": partial(TopK, act_fn=nn.Identity()),
    "TopKReLU": partial(TopK, act_fn=nn.ReLU()),
    "TopKELU1p": partial(TopK, act_fn=ELU1p()),
    "TopKabs": partial(TopK, use_abs=True, act_fn=nn.Identity()),
    "TopKabsReLU": partial(TopK, use_abs=True, act_fn=nn.ReLU()),
    "TopKabsELU1p": partial(TopK, use_abs=True, act_fn=ELU1p()),
    "BatchTopK": partial(BatchTopK, act_fn=nn.Identity()),
    "BatchTopKReLU": partial(BatchTopK, act_fn=nn.ReLU()),
    "BatchTopKELU1p": partial(BatchTopK, act_fn=ELU1p()),
    "JumpReLU": JumpReLU,
}

def get_activation(activation: str) -> nn.Module:
    """
    Get activation function by name, optionally with parameters.
    
    Handles string formats like:
    - "ReLU" -> nn.ReLU()
    - "TopK_100" -> TopK(k=100, act_fn=nn.Identity())
    
    Args:
        activation: Name of the activation, optionally with parameters
        
    Returns:
        Instantiated activation module
        
    Raises:
        ValueError: If activation name is not recognized
    """
    if "_" in activation:
        activation, arg = activation.split("_", maxsplit=1)
        if "TopK" in activation:
            return ACTIVATIONS_CLASSES[activation](k=int(arg))
        elif activation == "JumpReLU":
            return ACTIVATIONS_CLASSES[activation](hidden_dim=int(arg))
    return ACTIVATIONS_CLASSES[activation]()


class Autoencoder(nn.Module):
    """
    Sparse Autoencoder implementation.
    
    This implements the architecture:
        latents = activation(encoder(x - pre_bias) + latent_bias)
        recons = decoder(latents) + pre_bias
        
    Features:
    - Optional tied weights (decoder = encoder.T)
    - Tracking of neuron activation statistics
    - Re-initialization of dead neurons
    - Multiple weight initialization schemes
    - Soft capping to prevent exploding activations
    - Input normalization
    
    Args:
        n_latents: Dimension of the latent space
        n_inputs: Dimension of the input data
        activation: Activation function (string name or Module)
        tied: Whether to tie encoder and decoder weights
        normalize: Whether to normalize input data
        bias_init: Initial value for pre_bias
        init_method: Weight initialization method ('kaiming', 'xavier', 'uniform', 'normal')
        latent_soft_cap: Maximum value for latent activations (set to 0 to disable)
    """

    def __init__(
        self, 
        n_latents: int, 
        n_inputs: int, 
        activation: str | nn.Module = nn.ReLU(), 
        tied: bool = False, 
        normalize: bool = False,
        bias_init: torch.Tensor | float = 0.0, 
        init_method: str = "kaiming", 
        latent_soft_cap: float = 30.0, 
        *args: Any, 
        **kwargs: Any
    ) -> None:
        super().__init__()
        
        # Convert string activation to module if needed
        if isinstance(activation, str):
            activation = get_activation(activation)
        
        # Store configuration
        self.tied = tied
        self.n_latents = n_latents
        self.n_inputs = n_inputs
        self.init_method = init_method
        self.bias_init = bias_init
        self.normalize = normalize
        self.activation = activation
        
        # Initialize weights and biases
        self.pre_bias = nn.Parameter(
            torch.full((n_inputs,), bias_init) if isinstance(bias_init, float) else bias_init.clone()
        )
        self.encoder = nn.Parameter(torch.zeros((n_latents, n_inputs)).t())
        self.latent_bias = nn.Parameter(torch.zeros(n_latents))
        
        # For tied weights, decoder is derived from encoder
        if tied:
            self.register_parameter('decoder', None)
        else:
            self.decoder = nn.Parameter(torch.zeros((n_latents, n_inputs)))
        
        # Soft capping to prevent latent values from growing too large
        self.latent_soft_cap = SoftCapping(latent_soft_cap) if latent_soft_cap > 0 else nn.Identity()
        
        # Initialize weights
        self._init_weights()

        # Setup statistics tracking
        self.register_buffer(
            "latents_activation_frequency", torch.zeros(n_latents, dtype=torch.int64, requires_grad=False)
        )
        self.num_updates = 0
        self.dead_latents: list[int] = []


    def get_and_reset_stats(self) -> torch.Tensor:
        """
        Get neuron activation statistics and reset counters.
        
        Returns:
            Activation frequencies for each latent neuron (0-1 scale)
        """
        activations = self.latents_activation_frequency.detach().cpu().float() / self.num_updates
        self.latents_activation_frequency.zero_()
        self.num_updates = 0
        return activations
    
    @torch.no_grad()
    def _init_weights(self, norm=0.1, neuron_indices: list[int] | None = None) -> None:
        """
        Initialize weights for encoder and decoder.
        
        Args:
            norm: Target norm for initialized weights
            neuron_indices: Indices of neurons to reinitialize (None for all)
        
        Raises:
            ValueError: If init_method is invalid
        """
        valid_methods = ["kaiming", "xavier", "uniform", "normal"]
        if self.init_method not in valid_methods:
            raise ValueError(f"Invalid init_method: {self.init_method}. Choose from: {valid_methods}")

        # Get decoder reference (either tied to encoder or separate)
        if self.tied:
            decoder_weight = self.encoder.t()
        else:
            decoder_weight = self.decoder
        
        # Create new weights with requested initialization
        new_W_dec = torch.zeros_like(decoder_weight)
        
        if self.init_method == "kaiming":
            new_W_dec = nn.init.kaiming_uniform_(new_W_dec, nonlinearity='relu')
        elif self.init_method == "xavier":
            new_W_dec = nn.init.xavier_uniform_(new_W_dec, gain=nn.init.calculate_gain('relu'))
        elif self.init_method == "uniform":
            new_W_dec = nn.init.uniform_(new_W_dec, a=-1, b=1)
        elif self.init_method == "normal":
            new_W_dec = nn.init.normal_(new_W_dec)

        # Scale to target norm
        new_W_dec *= (norm / new_W_dec.norm(p=2, dim=-1, keepdim=True))
        
        # Create new latent biases (zeros)
        new_l_bias = torch.zeros_like(self.latent_bias)
        
        # Create encoder weights (transposed decoder weights)
        new_W_enc = new_W_dec.t().clone()

        # Update parameters, either all or only specified indices
        if neuron_indices is None:
            # Update all weights
            if not self.tied:
                self.decoder.data = new_W_dec
            self.encoder.data = new_W_enc
            self.latent_bias.data = new_l_bias
        else:
            # Update only specified neurons
            if not self.tied:
                self.decoder.data[neuron_indices] = new_W_dec[neuron_indices]
            self.encoder.data[:, neuron_indices] = new_W_enc[:, neuron_indices]
            self.latent_bias.data[neuron_indices] = new_l_bias[neuron_indices]

    @torch.no_grad()
    def project_grads_decode(self) -> None:
        """
        Project decoder gradients to enforce constraints.
        
        This ensures that each latent dimension's decoder weights 
        maintain a specific norm during training.
        """
        if self.tied:
            weights = self.encoder.data.T
            grad = self.encoder.grad.T
        else:
            weights = self.decoder.data
            grad = self.decoder.grad
            
        if grad is None:
            return

        # Project gradients to maintain unit norm constraint
        # Calculate component of gradient parallel to weights
        grad_proj = (grad * weights).sum(dim=-1, keepdim=True) * weights
        
        # Subtract parallel component from gradients
        if self.tied:
            self.encoder.grad -= grad_proj.T
        else:
            self.decoder.grad -= grad_proj

    @torch.no_grad()
    def scale_to_unit_norm(self) -> None:
        """
        Scale decoder weights to have unit norm.
        
        This enforces a unit norm constraint on each latent dimension's
        decoder weights, which aids in training stability.
        """
        eps = torch.finfo(self.encoder.dtype).eps
        
        if self.tied:
            # For tied weights, normalize encoder's columns
            norm = self.encoder.data.T.norm(p=2, dim=-1, keepdim=True) + eps
            self.encoder.data.T /= norm
        else:
            # For untied weights, normalize decoder's rows
            norm = self.decoder.data.norm(p=2, dim=-1, keepdim=True) + eps
            self.decoder.data /= norm
            
            # Compensate in encoder to maintain same reconstruction
            self.encoder.data *= norm.t()
            
        # Scale latent bias to compensate
        self.latent_bias.data *= norm.squeeze()

    def encode_pre_act(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute pre-activation latent values.
        
        Args:
            x: Input tensor of shape [batch_size, n_inputs]
            
        Returns:
            Pre-activation latent values
        """
        # Remove bias
        x_unbiased = x - self.pre_bias
        
        # Compute all latents
        latents_pre_act = x_unbiased @ self.encoder + self.latent_bias
        return latents_pre_act

    def preprocess(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Preprocess input data (normalization if enabled).
        
        Args:
            x: Input tensor
            
        Returns:
            Processed tensor and processing info dictionary
        """
        if not self.normalize:
            return x, {}
            
        x_normalized, mu, std = normalize_data(x)
        return x_normalized, {"mu": mu, "std": std}

    def encode(
        self, 
        x: torch.Tensor, 
        topk_number: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """
        Encode input data to latent representation.
        
        Args:
            x: Input tensor of shape [batch_size, n_inputs]
            topk_number: Optional number of top activations to keep (for inference)
            
        Returns:
            - Encoded latents (with activation)
            - Full encoded latents (without TopK sparsity for inference)
            - Preprocessing info
        """
        # Preprocess input
        x, info = self.preprocess(x)
        
        # Calculate pre-activations
        pre_encoded = self.encode_pre_act(x)
        
        # Apply activation for training path
        encoded = self.activation(pre_encoded)
        
        # Calculate full activations for evaluation
        if isinstance(self.activation, TopK):
            # For TopK, use the non-sparse version during evaluation
            full_encoded = self.activation.forward_eval(pre_encoded)
        else:
            # For other activations, full activation is the same
            full_encoded = encoded.clone()
        
        # Apply TopK sparsity for inference if requested
        if topk_number is not None:
            # Get top-k values and their indices
            _, indices = torch.topk(full_encoded, k=min(topk_number, full_encoded.shape[-1]), dim=-1)
            values = torch.gather(full_encoded, -1, indices)
            
            # Create sparse output with only top-k values
            full_encoded_sparse = torch.zeros_like(full_encoded)
            full_encoded_sparse.scatter_(-1, indices, values)
            full_encoded = full_encoded_sparse

        # Apply soft capping to prevent exploding values
        caped_encoded = self.latent_soft_cap(encoded)
        capped_full_encoded = self.latent_soft_cap(full_encoded)
        
        return caped_encoded, capped_full_encoded, info


    def decode(
        self, 
        latents: torch.Tensor, 
        info: dict[str, Any] | None = None
    ) -> torch.Tensor:
        """
        Decode latent representation to input space.
        
        Args:
            latents: Latent tensor of shape [batch_size, n_latents]
            info: Preprocessing info for denormalization
            
        Returns:
            Reconstructed input tensor
        """
        if info is None:
            info = {}
            
        # Apply decoder weights
        if self.tied:
            reconstructed = latents @ self.encoder.t() + self.pre_bias
        else:
            reconstructed = latents @ self.decoder + self.pre_bias
        
        # Denormalize if needed
        if self.normalize:
            if "std" not in info or "mu" not in info:
                raise ValueError("Normalization info missing from decode() call")
                
            reconstructed = reconstructed * info["std"] + info["mu"]
            
        return reconstructed

    @torch.no_grad()
    def update_latent_statistics(self, latents: torch.Tensor) -> None:
        """
        Update activation statistics for latent neurons.
        
        Args:
            latents: Activated latent tensor of shape [batch_size, n_latents]
        """
        batch_size = latents.shape[0]
        self.num_updates += batch_size
        
        # Count how many times each neuron was active in this batch
        current_activation_frequency = (latents != 0).to(torch.int64).sum(dim=0)
        self.latents_activation_frequency += current_activation_frequency
    
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape [batch_size, n_inputs]
            
        Returns:
            - Input tensor (potentially normalized)
            - Latent activations 
            - Reconstructed tensor
            - Full latent activations (without TopK sparsity)
        """
        # Encode to latent space
        latents, full_latents, info = self.encode(x)
        
        # Update activation statistics
        self.update_latent_statistics(latents)
        
        # Decode back to input space both with and without TopK sparsity
        reconstructed_full = self.decode(full_latents, info)
        reconstructed = self.decode(latents, info)
        if isinstance(self.activation, JumpReLU) and self.training:
            # For JumpReLU, apply custom training path
            latents = self.activation.forward_train(latents)
            
        return reconstructed, latents, reconstructed_full, full_latents


class MSAE(Autoencoder):
    """
    Matryoshka Sparse Autoencoder with multiple sparsity levels.
    
    Extends regular Sparse Autoencoder by using multiple TopK activations
    with different k values on the same latent space. This creates a nested
    representation where:
    - Lower k values provide more sparse, selective representations
    - Higher k values provide more complete, detailed representations
    
    The name "Matryoshka" comes from Russian nesting dolls, as smaller
    representations are nested within larger ones.
    
    Args:
        n_latents: Dimension of the autoencoder latent space
        n_inputs: Dimension of the input data
        activation: Base activation function (must be TopK-compatible)
        tied: Whether to tie encoder and decoder weights
        normalize: Whether to normalize input data
        bias_init: Initial value for pre_bias
        init_method: Weight initialization method
        latent_soft_cap: Maximum value for latent activations
        nesting_list: List of k values for TopK activations in ascending order
        relative_importance: Weights for each nesting level during training
    """
    def __init__(
        self, 
        n_latents: int, 
        n_inputs: int, 
        activation: str = "TopKReLU", 
        tied: bool = False, 
        normalize: bool = False,
        bias_init: torch.Tensor | float = 0.0,
        init_method: str = "kaiming", 
        latent_soft_cap: float = 30.0,
        sparsity_levels: list[int] = [16, 32], 
        sparsity_coef: list[float] | None = None, 
        *args: Any, 
        **kwargs: Any
    ) -> None:
        # Ensure sparsity levels list is sorted in ascending order
        self.sparsity_levels = sorted(sparsity_levels)
        
        # Set sparsity coefficients weights for each nesting level
        if sparsity_coef is None:
            self.sparsity_coef = [1.0] * len(sparsity_levels)
        else:
            self.sparsity_coef = sparsity_coef
            
        # Validate that sparsity_levels and sparsity_levels have the same length
        if len(self.sparsity_levels) != len(self.sparsity_coef):
            raise ValueError(
                f"Length mismatch: sparsity_levels ({len(self.sparsity_levels)}) and "
                f"sparsity_coef ({len(self.sparsity_coef)})"
            )

        # Ensure activation is TopK-compatible
        if "TopK" not in activation:
            warnings.warn(
                f"MSAE: activation '{activation}' is not a TopK activation. Changing to TopKReLU"
            )
            activation = "TopKReLU"

        # Initialize parent with base activation (smallest k)
        base_activation = f"{activation}_{self.sparsity_levels[0]}"
        super().__init__(
            n_latents, 
            n_inputs, 
            base_activation, 
            tied, 
            normalize, 
            bias_init, 
            init_method, 
            latent_soft_cap,
            *args,
            **kwargs
        )

        # Create all TopK activations with different k values
        self.activation = nn.ModuleList([
            get_activation(f"{activation}_{levels}") 
            for levels in self.sparsity_levels
        ])

    def encode(
        self, 
        x: torch.Tensor, 
        topk_number: int | None = None
    ) -> tuple[list[torch.Tensor], torch.Tensor, dict[str, Any]]:
        """
        Encode input data using multiple sparsity levels.
        
        Args:
            x: Input tensor of shape [batch_size, n_inputs]
            topk_number: Optional number of top activations to keep (overrides nesting)
            
        Returns:
            - List of encoded tensors at each sparsity level
            - Full encoded tensor (for inference)
            - Preprocessing info
        """
        # Preprocess input
        x, info = self.preprocess(x)
        
        # Calculate pre-activations
        pre_encoded = self.encode_pre_act(x)
        
        # Apply different TopK activations to create nested representations
        encoded = [activation(pre_encoded) for activation in self.activation]
        
        # Apply soft capping to each representation
        capped_encoded = [self.latent_soft_cap(enc) for enc in encoded]
        
        # For inference, apply custom topk if specified
        if topk_number is not None:
            # Use the densest representation as a starting point
            last_encoded = capped_encoded[-1]
            
            # Apply custom topk
            _, indices = torch.topk(last_encoded, k=min(topk_number, last_encoded.shape[-1]), dim=-1)
            values = torch.gather(last_encoded, -1, indices)
            
            # Create sparse representation
            custom_sparse = torch.zeros_like(last_encoded)
            custom_sparse.scatter_(-1, indices, values)
            capped_full_encoded = custom_sparse
        else:
            # Use the densest representation for reconstruction
            capped_full_encoded = capped_encoded[-1]
        
        return capped_encoded, capped_full_encoded, info

    def decode(
        self, 
        latents: list[torch.Tensor], 
        info: dict[str, Any] | None = None
    ) -> list[torch.Tensor]:
        """
        Decode latent representations at multiple sparsity levels.
        
        Args:
            latents: List of latent tensors at different sparsity levels
            info: Preprocessing info for denormalization
            
        Returns:
            List of reconstructed tensors at each sparsity level
        """
        if info is None:
            info = {}
        
        # Apply appropriate decoder weights
        if self.tied:
            # For tied weights, use encoder's transpose
            reconstructions = [
                latent @ self.encoder.t() + self.pre_bias 
                for latent in latents
            ]
        else:
            # For untied weights, use decoder
            reconstructions = [
                latent @ self.decoder + self.pre_bias 
                for latent in latents
            ]
        
        # Denormalize if needed
        if self.normalize:
            if info is None or "std" not in info or "mu" not in info:
                raise ValueError("Normalization info missing for decode operation")
                
            reconstructions = [
                recon * info["std"] + info["mu"] 
                for recon in reconstructions
            ]
            
        return reconstructions
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Matryoshka autoencoder (MSAE).
        
        Processes input through multiple sparsity levels.
        
        Args:
            x: Input tensor of shape [batch_size, n_inputs]
            
        Returns:
            - List of reconstructed tensors at each sparsity level
            - List of latent tensors at each sparsity level
            - Final reconstruction (at highest sparsity level)
            - Final latent representation (at highest sparsity level)
        """
        # Encode to latent space
        latents, full_latents, info = self.encode(x)
        
        # Update activation statistics
        sparsest_latents = latents[0]
        self.update_latent_statistics(sparsest_latents)
        
        # Decode back to input space both with and without TopK sparsity
        reconstructed = self.decode(latents, info)
        reconstructed_full = reconstructed[-1]
        
        return reconstructed, latents, reconstructed_full, full_latents


def load_model(path: str) -> tuple[Autoencoder, bool, bool, torch.Tensor]:
    """
    Load a saved autoencoder model from a path, inferring configuration from filename.
    
    Args:
        path: Path to saved model file
        
    Returns:
        - Loaded model
        - Whether data was mean-centered
        - Whether data was normalized
        - Scaling factor
    """
    # Extract model configuration from filename
    path_head = path.split("/")[-1]
    path_name = path_head[:path_head.find(".pt")]
    path_name_split = path_name.split("_")
    
    # Extract model architecture params
    n_latents = int(path_name_split.pop(0))
    n_inputs = int(path_name_split.pop(0))
    activation = path_name_split.pop(0)
    if "JumpReLU" in activation or "TopK" in activation:
        activation += "_" + path_name_split.pop(0)
    tied = True if "True" == path_name_split.pop(0) else False
    normalize = True if "True" == path_name_split.pop(0) else False
    latent_soft_cap = float(path_name_split.pop(0))
    
    model = Autoencoder(
        n_latents, 
        n_inputs, 
        activation, 
        tied=tied, 
        normalize=normalize, 
        latent_soft_cap=latent_soft_cap
    )

    # Load state dictionary
    model_state_dict = torch.load(
        path, 
        map_location='cuda' if torch.cuda.is_available() else 'cpu'
    )
    model.load_state_dict(model_state_dict['model'])
    dataset_normalize = model_state_dict['dataset_normalize']
    dataset_target_norm = model_state_dict['dataset_target_norm']
    dataset_mean = model_state_dict['dataset_mean']
    
    return model, dataset_normalize, dataset_target_norm, dataset_mean