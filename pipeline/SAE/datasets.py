import random
import torch.utils
from tqdm import tqdm
import torch
import numpy as np

from utils import set_logger, get_device

_, logger = set_logger(level="INFO")


class TunedLens(torch.nn.Module):
    """
    Tuned lens for sparse autoencoders.

    This module is used to adjust the lens of the model based on the input
    data and the specified configuration.

    Args:
        lens_shape: Shape of the lens
        use_bias : Whether to use bias in the lens
        use_residual : Whether to use residual connections
        eye_init: Whether to initialize the lens with an identity matrix
    """
    def __init__(self, lens_shape: int, use_bias: bool = True, use_residual: bool = True, eye_init: bool = False):
        super().__init__()
        self.lens = torch.nn.Linear(lens_shape, lens_shape, bias=use_bias)
        self.use_residual = use_residual
        torch.nn.init.eye_(self.lens.weight) if eye_init else self.lens.weight.data.zero_()
        if use_bias:
            self.lens.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lens(x) + x if self.use_residual else self.lens(x)

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        if self.lens.bias is not None:
            x = x - self.lens.bias
        identity = torch.eye(self.lens.weight.shape[0], device=x.device)
        transform_matrix = identity + self.lens.weight
        transform_matrix_inv = torch.inverse(transform_matrix)
        return torch.matmul(x, transform_matrix_inv)


class SAEDataset(torch.utils.data.Dataset):
    """
    Memory-efficient dataset implementation for Sparse Autoencoders.
    
    Uses memory-mapped arrays to efficiently load large datasets without
    loading the entire dataset into memory at once.
    
    Args:
        data_path: Path to the data file (expects naming format with size and dims at end)
        dtype: Torch data type to use for tensors
        mean_center: Whether to center the data around mean
        normalize: Whether to normalize vectors to unit length
        target_norm: Target norm for scaling (sqrt(dims) if None)
    """
    def __init__(
        self, 
        data_path: str, 
        dtype: torch.dtype = torch.float32, 
        mean_center: bool = False,
        normalize: bool = False, 
        target_norm: float | None = None
    ):

        # Quick metadata parsing
        parts = data_path.split("/")[-1].split(".")[0].split("_")
        self.len, self.vector_size = map(int, parts[-2:])
        
        # Set core attributes
        self.dtype = dtype
        self.data = np.memmap(data_path, dtype="float32", mode="r", 
                             shape=(self.len, self.vector_size))
        
        # Handle repr case efficiently
        if "repr" in data_path:
            self.mean = torch.zeros(self.vector_size, dtype=dtype)
            self.mean_center = self.normalize = False
            self.scaling_factor = 1.0
            return

        # Set configuration
        self.mean_center = mean_center
        self.normalize = normalize
        self.target_norm = np.sqrt(self.vector_size) if target_norm is None else target_norm

        # Compute statistics efficiently
        if self.mean_center or self.target_norm != 0.0:
            self._compute_statistics()
        else:
            self.mean = torch.zeros(self.vector_size, dtype=dtype)
            self.scaling_factor = 1.0

    def _compute_statistics(self, batch_size: int = 10000) -> None:
        """Compute mean and scaling factor in batches."""
        # Compute mean if needed
        if self.mean_center:
            mean_acc = np.zeros(self.vector_size, dtype=np.float32)
            total = 0

            for start in range(0, self.len, batch_size):
                end = min(start + batch_size, self.len)
                batch = self.data[start:end].copy()
                batch_size_actual = end - start
                
                if self.normalize:
                    norms = np.linalg.norm(batch, axis=1, keepdims=True)
                    np.divide(batch, norms, out=batch)
                
                mean_acc += np.sum(batch, axis=0)
                total += batch_size_actual

            self.mean = torch.from_numpy(mean_acc / total).to(self.dtype)
        else:
            self.mean = torch.zeros(self.vector_size, dtype=self.dtype)

        # Compute scaling factor if needed
        if self.target_norm != 0.0:
            squared_norm_sum = 0.0
            total = 0

            for start in range(0, self.len, batch_size):
                end = min(start + batch_size, self.len)
                batch = self.data[start:end].copy()
                batch_size_actual = end - start
                
                if self.normalize:
                    norms = np.linalg.norm(batch, axis=1, keepdims=True)
                    np.divide(batch, norms, out=batch)
                
                if self.mean_center:
                    batch = batch - self.mean.numpy()

                squared_norm_sum += np.sum(np.square(batch))
                total += batch_size_actual

            avg_squared_norm = squared_norm_sum / total
            self.scaling_factor = float(self.target_norm / avg_squared_norm)
        else:
            self.scaling_factor = 1.0

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return self.len

    def process_data(self, data: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Process data into the correct format."""
        X = torch.from_numpy(data).to(self.dtype)
        
        if self.normalize:
            norm = torch.norm(X, dim=-1, keepdim=True)
            X.div_(norm)
        
        if self.mean_center:
            X.sub_(self.mean)
            if self.normalize:
                norm = torch.norm(X, dim=-1, keepdim=True)
                X.div_(norm)
        
        if self.scaling_factor != 1.0:
            X.mul_(self.scaling_factor)
        
        return X
    
    def invert_preprocess(self, data: torch.Tensor, idx: int | None=None) -> torch.Tensor:
        """Inverse process data."""
        if self.scaling_factor != 1.0:
            data.div_(self.scaling_factor)

        if self.mean_center:
            data.add_(self.mean)

        return data

    @torch.no_grad()
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Optimized item retrieval."""
        return self.process_data(self.data[idx].copy())


class SDSAEDataset(torch.utils.data.Dataset):
    """
    Memory-efficient dataset implementation for Sparse Autoencoders.

    Uses memory-mapped arrays to efficiently load large datasets without
    loading the entire dataset into memory at once.

    Args:
        data_path: Path to the data file (expects naming format with size and dims at end)
        dtype: Torch data type to use for tensors
        mean_center: Whether to center the data around mean
        normalize: Whether to normalize vectors to unit length
        target_norm: Target norm for scaling (sqrt(dims) if None)
    """
    def __init__(
        self,
        data_path: str,
        seq_len: int | None = None,
        use_lens: bool = False,
        seq_id: int | None = None,
        dtype: torch.dtype = torch.float32,
        mean_center: bool = False,
        normalize: bool = False,
        target_norm: float | None = None,
        device: torch.device | None = None,
        hyper_search: bool = False
    ):
        self.lenses = None
        self.lenses_loss = None
        self.seq_len = seq_len
        self.use_lens = use_lens
        self.seq_id = seq_id
        self.device = device if device is not None else get_device()
        self.hyper_search = hyper_search

        # Quick metadata parsing
        parts = data_path.split("/")[-1].split(".")[0].split("_")
        self.len, self.vector_size = map(int, parts[-2:])

        # Set core attributes
        self.dtype = dtype
        self.data = np.memmap(data_path, dtype="float32", mode="r",
                             shape=(self.len, self.vector_size))

        # Handle repr case efficiently
        if "repr" in data_path:
            self.mean = torch.zeros(self.vector_size, dtype=dtype)
            self.mean_center = self.normalize = False
            self.scaling_factor = 1.0
            return

        # Set configuration
        self.mean_center = mean_center
        self.normalize = normalize
        self.target_norm = np.sqrt(self.vector_size) if target_norm is None else target_norm

        # Compute statistics efficiently
        if self.mean_center or self.target_norm != 0.0:
            self._compute_statistics()
        else:
            self.mean = torch.zeros(self.vector_size, dtype=dtype)
            self.scaling_factor = 1.0

        # Prepare lens if needed)
        if self.use_lens:
            logger.info(f"Preparing {self.seq_len-1} lenses for {self.seq_len} sequences")
            self.lenses = {}
            self.lenses_loss = {}
            for key in range(self.seq_len):
                if key == self.seq_id:
                    continue
                lens, loss = self._prepare_lens(key)
                self.lenses[key] = lens
                self.lenses_loss[key] = loss

            logger.info(f"All {self.seq_len-1} lenses prepared")


    def _prepare_lens(self, key: int) -> tuple[TunedLens, float]:
        """
        Prepares and trains a TunedLens model for a specific key.
        This method initializes a TunedLens model, prepares the input and target datasets
        based on the provided key, and trains the model using Mean Squared Error (MSE) loss.
        The trained model and the final loss value are returned.
        Args:
            key (int): The key used to determine the input and target indices for training.
        Returns:
            tuple[TunedLens, float]: A tuple containing the trained TunedLens model and the final loss value.
        Raises:
            ValueError: If the generated input or target indices are empty.
        Notes:
            - The method uses a batch size of 1024 and trains the model for 10 epochs.
            - The AdamW optimizer is used with a learning rate of 0.005.
            - Gradient clipping is applied with a maximum norm of 1.0.
            - The model is trained on a GPU if available, otherwise on a CPU.
        """
        # Create proper indices for selection
        input_indices = list(range(key, self.len, self.seq_len))
        target_indices = list(range(self.seq_id, self.len, self.seq_len))

        # Ensure we're getting data
        if not input_indices or not target_indices:
            raise ValueError(f"Empty indices list: input={len(input_indices)}, target={len(target_indices)}")

        # Get data with proper indices
        input_data = self.data[input_indices]
        target_data = self.data[target_indices]

        input_dataset = [self.process_data(x) for x in input_data]
        target_dataset = [self.process_data(x) for x in target_data]

        final_dataset = torch.utils.data.TensorDataset(torch.stack(input_dataset), torch.stack(target_dataset))
        train_dataset, test_dataset = torch.utils.data.random_split(final_dataset, [0.8, 0.2])
        
        if self.hyper_search:
            use_optuna = False
            try:
                import optuna
                use_optuna = True
            except ImportError:
                logger.warning("Optuna not installed, proceeding without hyperparameter optimization.")
        else:
            use_optuna = False
        
        if use_optuna:
            def objective(trial):
                # Define hyperparameters to search
                lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
                batch_size = trial.suggest_categorical('batch_size', [512, 1024, 2048, 4096])
                n_epochs = trial.suggest_int('n_epochs', 1, 20)
                use_bias = trial.suggest_categorical('use_bias', [True, False])
                # use_residual = trial.suggest_categorical('use_residual', [True, False])
                use_residual = True
                eye_init = trial.suggest_categorical('eye_init', [True, False])
                use_gradient_clipping = trial.suggest_categorical('use_gradient_clipping', [True, False])
                
                # Create model with trial params
                lens = TunedLens(self.vector_size, use_bias=use_bias, use_residual=use_residual, eye_init=eye_init)
                
                # Training data
                train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
                test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
                
                # Training setup with trial params
                criterion = torch.nn.MSELoss()
                optimizer = torch.optim.AdamW(lens.parameters(), lr=lr)
                
                lens.to(self.device)
                lens.train()
                
                # Training loop
                final_loss = 0
                for epoch in range(n_epochs):
                    for input_data, target_data in train_dl:
                        input_data = input_data.to(self.device)
                        target_data = target_data.to(self.device)
                        
                        optimizer.zero_grad()
                        output = lens(input_data)
                        loss = criterion(output, target_data)
                        loss.backward()
                        if use_gradient_clipping:
                            torch.nn.utils.clip_grad_norm_(lens.parameters(), 1.0)
                        optimizer.step()
                    
                    final_loss = 0
                    for input_data, target_data in test_dl:
                        input_data = input_data.to(self.device)
                        target_data = target_data.to(self.device)
                        
                        output = lens(input_data)
                        loss = criterion(output, target_data)
                        final_loss += loss.item()
                    final_loss /= len(test_dl)
                    
                # Clean up
                del train_dl, test_dl, optimizer, criterion
                
                return final_loss  # Return the metric to optimize

            # Create and run the study
            logger.info("Starting hyperparameter optimization with Optuna")
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=50)

            # Get best parameters
            best_params = study.best_params
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best loss: {study.best_value}")
            # Create model with best parameters
            use_bias = best_params['use_bias']
            # use_residual = best_params['use_residual']
            use_residual = True
            eye_init = best_params['eye_init']
            batch_size = best_params['batch_size']
            n_epochs = best_params['n_epochs']
            lr = best_params['lr']
            use_gradient_clipping = best_params['use_gradient_clipping']
        else:
            # Use default parameters
            batch_size = 1024
            n_epochs = 15 #10
            use_bias = True
            use_residual = True
            eye_init = False
            lr = 0.005
            use_gradient_clipping = True
                    
        
        lens = TunedLens(self.vector_size, use_bias=use_bias, use_residual=use_residual, eye_init=eye_init)

        train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(lens.parameters(), lr=lr)

        lens.to(self.device)
        lens.train()

        logger.info(f"Training lens {key} with {len(input_dataset)} samples")
        for epoch in range(n_epochs):
            for input_data, target_data in tqdm(train_dl, desc=f"Training lens {key}"):
                input_data = input_data.to(self.device)
                target_data = target_data.to(self.device)

                optimizer.zero_grad()
                output = lens(input_data)
                loss = criterion(output, target_data)
                loss.backward()
                if use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(lens.parameters(), 1.0)
                optimizer.step()
            
            eval_loss = 0
            for input_data, target_data in test_dl:
                input_data = input_data.to(self.device)
                target_data = target_data.to(self.device)

                output = lens(input_data)
                eval_loss_sub = criterion(output, target_data)
                eval_loss += eval_loss_sub.item()
            eval_loss /= len(test_dl)
            
            logger.info(f"Epoch {epoch+1}/{n_epochs} - Lens {key} Train Loss: {loss.item():.4f} Eval Loss: {eval_loss:.4f}")
            if eval_loss < 0.0005:
                break
        
        # Clean up
        del train_dl, test_dl, optimizer, criterion
        logger.info(f"Lens {key} trained with final loss: {loss.item():.4f} Eval Loss: {eval_loss:.4f}")
        lens.eval()
        lens.to(torch.device("cpu"))
        return lens, eval_loss


    def _compute_statistics(self, batch_size: int = 10000) -> None:
        """Compute mean and scaling factor in batches."""
        # Compute mean if needed
        if self.mean_center:
            mean_acc = np.zeros(self.vector_size, dtype=np.float32)
            total = 0

            for start in range(0, self.len, batch_size):
                end = min(start + batch_size, self.len)
                batch = self.data[start:end].copy()
                batch_size_actual = end - start

                if self.normalize:
                    norms = np.linalg.norm(batch, axis=1, keepdims=True)
                    np.divide(batch, norms, out=batch)

                mean_acc += np.sum(batch, axis=0)
                total += batch_size_actual

            self.mean = torch.from_numpy(mean_acc / total).to(self.dtype)
        else:
            self.mean = torch.zeros(self.vector_size, dtype=self.dtype)

        # Compute scaling factor if needed
        if self.target_norm != 0.0:
            squared_norm_sum = 0.0
            total = 0

            for start in range(0, self.len, batch_size):
                end = min(start + batch_size, self.len)
                batch = self.data[start:end].copy()
                batch_size_actual = end - start

                if self.normalize:
                    norms = np.linalg.norm(batch, axis=1, keepdims=True)
                    np.divide(batch, norms, out=batch)

                if self.mean_center:
                    batch = batch - self.mean.numpy()

                squared_norm_sum += np.sum(np.square(batch))
                total += batch_size_actual

            avg_squared_norm = squared_norm_sum / total
            self.scaling_factor = float(self.target_norm / avg_squared_norm)
        else:
            self.scaling_factor = 1.0

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return self.len

    def process_data(self, data: np.ndarray | torch.Tensor, idx: int | None=None) -> torch.Tensor:
        """Process data into the correct format."""
        X = torch.from_numpy(data).to(self.dtype)

        if self.normalize:
            norm = torch.norm(X, dim=-1, keepdim=True)
            X.div_(norm)

        if self.mean_center:
            X.sub_(self.mean)
            if self.normalize:
                norm = torch.norm(X, dim=-1, keepdim=True)
                X.div_(norm)

        if self.scaling_factor != 1.0:
            X.mul_(self.scaling_factor)

        if self.lenses is not None and idx is not None:
            current_seq_id = idx % self.seq_len
            if current_seq_id != self.seq_id:
                lens = self.lenses[current_seq_id]
                X = lens(X)

        return X

    def invert_preprocess(self, data: torch.Tensor, idx: int | None=None) -> torch.Tensor:
        """Inverse process data."""
        if self.lenses is not None and idx is not None:
            current_seq_id = idx % self.seq_len
            if current_seq_id != self.seq_id:
                lens = self.lenses[current_seq_id]
                X = lens.invert(data)
            else:
                X = data
        else:
            X = data

        if self.scaling_factor != 1.0:
            X.div_(self.scaling_factor)

        if self.mean_center:
            X.add_(self.mean)

        return X

    @torch.no_grad()
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Optimized item retrieval."""
        return self.process_data(self.data[idx].copy(), idx)


class MultiSAEDataset(torch.utils.data.Dataset):
    """
    Memory-efficient dataset implementation for Sparse Autoencoders.

    Uses memory-mapped arrays to efficiently load large datasets without
    loading the entire dataset into memory at once.

    Args:
        data_path_first: Path to the data file for first modality (expects naming format with size and dims at end)
        data_path_second: Path to the data file for second modality (expects naming format with size and dims at end)
        dtype: Torch data type to use for tensors
        mean_center: Whether to center the data around mean
        normalize: Whether to normalize vectors to unit length
        target_norm: Target norm for scaling (sqrt(dims) if None)
    """
    def __init__(
        self,
        data_path_first: str,
        data_path_second: str,
        use_lens: bool = False,
        use_first: bool = True,
        dtype: torch.dtype = torch.float32,
        mean_center: bool = False,
        normalize: bool = False,
        target_norm: float | None = None,
        device: torch.device | None = None,
        hyper_search: bool = False
    ):
        self.lenses = None
        self.lenses_loss = 0.0
        self.use_lens = use_lens
        self.use_first = use_first
        self.dtype = dtype
        self.device = device
        self.hyper_search = hyper_search

        # Quick metadata parsing for first modality
        parts_first = data_path_first.split("/")[-1].split(".")[0].split("_")
        len_first, vector_size_first = map(int, parts_first[-2:])

        # Set core attributes
        self.data_first = np.memmap(data_path_first, dtype="float32", mode="r",
                             shape=(len_first, vector_size_first))

        # Quick metadata parsing for second modality
        parts_second = data_path_second.split("/")[-1].split(".")[0].split("_")
        len_second, vector_size_second = map(int, parts_second[-2:])

        # Set core attributes
        self.data_second = np.memmap(data_path_second, dtype="float32", mode="r",
                             shape=(len_second, vector_size_second))

        assert vector_size_first == vector_size_second, "Vector size of first and second modality must match"
        self.vector_size = vector_size_first

        if len_first != len_second:
            logger.warning(
                f"Length of first modality {len_first} does not match second modality {len_second}. "
                f"Using length of the shorter one."
            )
            self.len = min(len_first, len_second)
        else:
            self.len = len_first

        # Set configuration
        self.mean_center = mean_center
        self.normalize = normalize
        self.target_norm = np.sqrt(self.vector_size) if target_norm is None else target_norm

        # Compute statistics efficiently
        if self.mean_center or self.target_norm != 0.0:
            self.mean_first, self.scaling_factor_first = self._compute_statistics(self.data_first)
            self.mean_second, self.scaling_factor_second = self._compute_statistics(self.data_second)
        else:
            self.mean_first = torch.zeros(self.vector_size, dtype=dtype)
            self.mean_second = torch.zeros(self.vector_size, dtype=dtype)
            self.scaling_factor_first, self.scaling_factor_second = 1.0, 1.0

        # Prepare lens if needed)
        if self.use_lens:
            logger.info(f"Preparing lenses for second modality {self.use_first}")
            self.lenses, self.lenses_loss = self._prepare_lens()
            logger.info("Lens prepared")

    def _prepare_lens(self) -> tuple[TunedLens, float]:
        """
        Prepares and trains a TunedLens model for a specific key.
        This method initializes a TunedLens model, prepares the input and target datasets
        based on the provided key, and trains the model using Mean Squared Error (MSE) loss.
        The trained model and the final loss value are returned.
        Args:
            key (int): The key used to determine the input and target indices for training.
        Returns:
            tuple[TunedLens, float]: A tuple containing the trained TunedLens model and the final loss value.
        Raises:
            ValueError: If the generated input or target indices are empty.
        Notes:
            - The method uses a batch size of 1024 and trains the model for 10 epochs.
            - The AdamW optimizer is used with a learning rate of 0.005.
            - Gradient clipping is applied with a maximum norm of 1.0.
            - The model is trained on a GPU if available, otherwise on a CPU.
        """
        class LensDataset(torch.utils.data.Dataset):
            def __init__(self, data_first: np.ndarray, data_second: np.ndarray, use_first: bool, transform: callable):
                self.data_first = data_first
                self.data_second = data_second
                self.use_first = use_first
                self.transform = transform

            def __len__(self):
                return len(self.data_first)

            def __getitem__(self, idx):
                first = self.transform(self.data_first[idx].copy(), True)
                second = self.transform(self.data_second[idx].copy(), False)

                if self.use_first:
                    target = first
                    input = second
                else:
                    target = second
                    input = first
                return input, target

        if self.hyper_search:
            use_optuna = False
            try:
                import optuna
                use_optuna = True
            except ImportError:
                logger.warning("Optuna not installed, proceeding without hyperparameter optimization.")
        else:
            use_optuna = False
        
        final_dataset = LensDataset(
            self.data_first, self.data_second, self.use_first, self.process_data
        )
        train_dataset, test_dataset = torch.utils.data.random_split(final_dataset, [0.8, 0.2])
        
        if use_optuna:
            def objective(trial):
                # Define hyperparameters to search
                lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
                batch_size = trial.suggest_categorical('batch_size', [512, 1024, 2048, 4096])
                n_epochs = trial.suggest_int('n_epochs', 1, 20)
                use_bias = trial.suggest_categorical('use_bias', [True, False])
                # use_residual = trial.suggest_categorical('use_residual', [True, False])
                use_residual = True
                eye_init = trial.suggest_categorical('eye_init', [True, False])
                use_gradient_clipping = trial.suggest_categorical('use_gradient_clipping', [True, False])
                
                # Create model with trial params
                lens = TunedLens(self.vector_size, use_bias=use_bias, use_residual=use_residual, eye_init=eye_init)
                
                # Training data
                train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
                test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
                
                # Training setup with trial params
                criterion = torch.nn.MSELoss()
                optimizer = torch.optim.AdamW(lens.parameters(), lr=lr)
                
                lens.to(self.device)
                lens.train()
                
                # Training loop
                final_loss = 0
                for epoch in range(n_epochs):
                    for input_data, target_data in train_dl:
                        input_data = input_data.to(self.device)
                        target_data = target_data.to(self.device)
                        
                        optimizer.zero_grad()
                        output = lens(input_data)
                        loss = criterion(output, target_data)
                        loss.backward()
                        if use_gradient_clipping:
                            torch.nn.utils.clip_grad_norm_(lens.parameters(), 1.0)
                        optimizer.step()
                    
                    final_loss = 0
                    for input_data, target_data in test_dl:
                        input_data = input_data.to(self.device)
                        target_data = target_data.to(self.device)
                        
                        output = lens(input_data)
                        loss = criterion(output, target_data)
                        final_loss += loss.item()
                    final_loss /= len(test_dl)
                
                # Clean up
                del train_dl, test_dl, optimizer, criterion
                
                return final_loss  # Return the metric to optimize

            # Create and run the study
            logger.info("Starting hyperparameter optimization with Optuna")
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=50)

            # Get best parameters
            best_params = study.best_params
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best loss: {study.best_value}")
            # Create model with best parameters
            use_bias = best_params['use_bias']
            # use_residual = best_params['use_residual']
            use_residual = True
            eye_init = best_params['eye_init']
            batch_size = best_params['batch_size']
            n_epochs = best_params['n_epochs']
            lr = best_params['lr']
            use_gradient_clipping = best_params['use_gradient_clipping']
        else:
            use_bias = True
            use_residual = True
            eye_init = False
            batch_size = 1024
            n_epochs = 2 #5 #10
            lr = 0.005
            use_gradient_clipping = True
            
        lens = TunedLens(self.vector_size, use_bias=use_bias, use_residual=use_residual, eye_init=eye_init)

        train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(lens.parameters(), lr=lr)

        lens.to(self.device)
        lens.train()

        logger.info(f"Training lens with {len(final_dataset)} samples")
        for epoch in range(n_epochs):
            for input_data, target_data in tqdm(train_dl, desc="Training lens"):
                input_data = input_data.to(self.device)
                target_data = target_data.to(self.device)

                optimizer.zero_grad()
                output = lens(input_data)
                loss = criterion(output, target_data)
                loss.backward()
                if use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(lens.parameters(), 1.0)
                optimizer.step()

            eval_loss = 0
            for input_data, target_data in test_dl:
                input_data = input_data.to(self.device)
                target_data = target_data.to(self.device)

                output = lens(input_data)
                loss = criterion(output, target_data)
                eval_loss += loss.item()
            eval_loss /= len(test_dl)

            logger.info(f"Epoch {epoch+1}/{n_epochs} - Lens Train Loss: {loss.item():.4f} Eval Loss: {eval_loss:.4f}")
            if eval_loss < 0.0005:
                break

        del train_dl, test_dl, optimizer, criterion
        logger.info(f"Lens trained with final loss: {loss.item():.4f} Eval Loss: {eval_loss:.4f}")
        lens.eval()
        lens.to(torch.device("cpu"))
        return lens, eval_loss


    def _compute_statistics(self, data: np.ndarray, batch_size: int = 10000) -> tuple[np.ndarray, float]:
        """Compute mean and scaling factor in batches."""
        # Compute mean if needed
        if self.mean_center:
            mean_acc = np.zeros(self.vector_size, dtype=np.float32)
            total = 0

            for start in range(0, self.len, batch_size):
                end = min(start + batch_size, self.len)
                batch = data[start:end].copy()
                batch_size_actual = end - start

                if self.normalize:
                    norms = np.linalg.norm(batch, axis=1, keepdims=True)
                    np.divide(batch, norms, out=batch)

                mean_acc += np.sum(batch, axis=0)
                total += batch_size_actual

            mean = torch.from_numpy(mean_acc / total).to(self.dtype)
        else:
            mean = torch.zeros(self.vector_size, dtype=self.dtype)

        # Compute scaling factor if needed
        if self.target_norm != 0.0:
            squared_norm_sum = 0.0
            total = 0

            for start in range(0, self.len, batch_size):
                end = min(start + batch_size, self.len)
                batch = data[start:end].copy()
                batch_size_actual = end - start

                if self.normalize:
                    norms = np.linalg.norm(batch, axis=1, keepdims=True)
                    np.divide(batch, norms, out=batch)

                if self.mean_center:
                    batch = batch - mean.numpy()

                squared_norm_sum += np.sum(np.square(batch))
                total += batch_size_actual

            avg_squared_norm = squared_norm_sum / total
            scaling_factor = float(self.target_norm / avg_squared_norm)
        else:
            scaling_factor = 1.0

        return mean, scaling_factor

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return self.len

    def process_data(self, data: np.ndarray | torch.Tensor, first_modality: bool=True) -> torch.Tensor:
        """Process data into the correct format."""
        X = torch.from_numpy(data).to(self.dtype)
        mean = self.mean_first if first_modality else self.mean_second
        scaling_factor = self.scaling_factor_first if first_modality else self.scaling_factor_second

        if self.normalize:
            norm = torch.norm(X, dim=-1, keepdim=True)
            X.div_(norm)

        if self.mean_center:
            X.sub_(mean)
            if self.normalize:
                norm = torch.norm(X, dim=-1, keepdim=True)
                X.div_(norm)

        if scaling_factor != 1.0:
            X.mul_(scaling_factor)

        if self.lenses and self.use_first != first_modality:
            X = self.lenses(X)

        return X

    def invert_preprocess(self, data: torch.Tensor, first_modality: bool=True) -> torch.Tensor:
        """Inverse process data."""
        mean = self.mean_first if first_modality else self.mean_second
        scaling_factor = self.scaling_factor_first if first_modality else self.scaling_factor_second

        if self.lenses and self.use_first != first_modality:
            data = self.lenses.invert(data)

        if scaling_factor != 1.0:
            data.div_(scaling_factor)

        if self.mean_center:
            data.add_(mean)

        return data


    @torch.no_grad()
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Optimized item retrieval."""
        return self.process_data(self.data_first[idx].copy(), True), self.process_data(self.data_second[idx].copy(), False)


class SimMultiSAEDataset(torch.utils.data.Dataset):
    def __init__(self, data: MultiSAEDataset):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        first, second = self.data[index]
        output = first if random.random() < 0.5 else second
        return output