import functools
import torch
import dataclasses
import math # Included just in case, torch functions are generally sufficient

# Use dataclass for clean configuration and make it immutable/hashable
@dataclasses.dataclass(frozen=True, eq=True)
class CFG:
    """
    Configuration class for diffusion parameters.

    Uses dataclasses for clean initialization, immutability (frozen=True),
    and value-based hashing/equality (eq=True) which is good for caching
    if the CFG object itself were used directly as a key (though here it's
    an instance attribute).
    """
    kappa: float = 2
    p: float = 0.3
    eta_T: float = 0.999
    T: int = 10 #ll be calculated in post_init, not passed during initialization
    eta_1: float = dataclasses.field(init=False)

    def __post_init__(self):
        """Calculates dependent parameters and performs validation after initialization."""
        if not isinstance(self.T, int) or self.T <= 1:
            raise ValueError(f"Time steps T must be an integer greater than 1, but got {self.T}.")
        if not isinstance(self.kappa, (int, float)) or self.kappa <= 0:
            raise ValueError(f"kappa must be a positive number, but got {self.kappa}.")
        if not isinstance(self.p, (int, float)) or self.p < 0:
            raise ValueError(f"p must be a non-negative number, but got {self.p}.")
        if not isinstance(self.eta_T, (int, float)) or not (0 < self.eta_T <= 1):
            raise ValueError(f"eta_T must be a number between 0 and 1 (inclusive of 1), but got {self.eta_T}.")

        # Calculate eta_1. Since frozen=True, use object.__setattr__.
        calculated_eta_1 = min(0.001, (0.04 / self.kappa)**2)
        object.__setattr__(self, 'eta_1', calculated_eta_1)

        if not isinstance(self.eta_1, (int, float)) or self.eta_1 <= 0:
             # This shouldn't happen if kappa is positive, but good defensive check
             raise ValueError(f"Calculated eta_1 is not a positive number: {self.eta_1}")


class DiffusionScheduler:
    """
    Manages diffusion parameters and noise generation based on a given configuration.

    Encapsulates parameter calculation logic and uses caching for efficiency.
    """
    def __init__(self, config: CFG):
        """
        Initializes the DiffusionScheduler with a configuration.

        Args:
            config: An instance of the CFG configuration class.
        """
        if not isinstance(config, CFG):
            raise TypeError("config must be an instance of CFG")
        self.config = config
        # You could pre-calculate some values here if desired, but lru_cache handles lazy calculation

    @functools.lru_cache(maxsize=1) # b_0 only depends on config, so only one entry needed per instance
    def _get_b_0(self) -> torch.Tensor:
        """
        Calculates and returns the value of b_0.
        Cached per instance.
        """
        # Denominator 2*T - 2 is guaranteed > 0 by CFG validation (T>1)
        log_ratio = torch.log(torch.tensor(self.config.eta_T / self.config.eta_1))
        # Ensure the calculation is done with floats to avoid integer division issues
        return torch.exp(log_ratio / (2 * torch.tensor(self.config.T - 1, dtype=torch.float32)))

    @functools.lru_cache(maxsize=None) # Cache all results for t=0 to T
    def get_beta_t(self, t: int) -> torch.Tensor:
        """
        Calculates and returns the value of beta_t for time step t.

        Defines beta_0 = 0 for t=0 as a base case.
        For t >= 1, uses the provided formula.
        Cached per instance and time step t.
        """
        if t == 0:
            return torch.tensor(0.0)
        if t < 0:
             raise ValueError(f"Time step t cannot be negative for beta_t calculation, but got {t}.")
        if t > self.config.T:
            # While beta_t could be calculated for t > T, in diffusion context t is usually 0..T
             print(f"Warning: Calculating beta_t for t={t} which is > config.T={self.config.T}.")
             # We will allow it for formula consistency, but it might indicate incorrect usage.

        # Formula for t >= 1
        # Use float for division and power
        numerator = torch.tensor(t - 1, dtype=torch.float32)
        denominator = torch.tensor(self.config.T - 1, dtype=torch.float32) # Denom is > 0 by CFG validation

        # torch.pow handles 0**p correctly (0 for p>0, 1 for p=0)
        normalized_time_pow_p = torch.pow(numerator / denominator, self.config.p)

        # Scale by (T-1).
        return normalized_time_pow_p * (self.config.T - 1)

    @functools.lru_cache(maxsize=None) # Cache all results for t=0 to T
    def get_eta_t(self, t: int) -> torch.Tensor:
        """
        Calculates and returns the value of eta_t for time step t.

        Defines eta_0 = eta_1 for t=0 as a base case.
        For t >= 1, uses the provided formula.
        Cached per instance and time step t.
        """
        if t == 0:
            # Base case definition, assuming eta_0 = eta_1
            return torch.tensor(self.config.eta_1)
        if t < 0:
             raise ValueError(f"Time step t cannot be negative for eta_t calculation, but got {t}.")
        if t > self.config.T:
             print(f"Warning: Calculating eta_t for t={t} which is > config.T={self.config.T}.")
             # We will allow it for formula consistency, but it might indicate incorrect usage.


        # For t >= 1
        b_0 = self._get_b_0()       # Uses cached result
        beta_t = self.get_beta_t(t) # Uses cached result

        # eta_1 > 0 and b_0 > 0 and beta_t >= 0 means eta_t should be >= eta_1 > 0
        # Use torch.tensor() around config.eta_1 to ensure it's a tensor for multiplication
        return torch.tensor(self.config.eta_1) * torch.pow(b_0, beta_t * 2)

    @functools.lru_cache(maxsize=None) # Cache all results for t=1 to T
    def get_alpha_t(self, t: int) -> torch.Tensor:
        """
        Calculates and returns the value of alpha_t, defined as beta_t - beta_{t-1}.

        This parameter is typically used for t=1 to T.
        Requires t >= 1.
        Cached per instance and time step t.
        """
        if t < 1:
            # alpha_t is a difference, needs at least t=1 to use t-1=0
            raise ValueError(f"alpha_t is defined for t >= 1, but received t={t}.")
        if t > self.config.T:
             print(f"Warning: Calculating alpha_t for t={t} which is > config.T={self.config.T}.")
             # We will allow it for formula consistency, but it might indicate incorrect usage.
        if t == 1:
            return self.get_eta_t(t)

        # get_beta_t handles the t-1=0 case internally
        return self.get_eta_t(t) - self.get_eta_t(t - 1)

    @functools.lru_cache(maxsize=None) # Cache all results for t=1 to T
    def get_loss_coef(self, t: int) -> torch.Tensor:
        """
        Calculates and returns the value of the loss coefficient at time t.

        Requires t >= 1 as it depends on eta_t and eta_{t-1}.
        Cached per instance and time step t.
        """
        if t < 1:
            # Loss coefficient is typically defined for t=1 to T
            raise ValueError(f"loss_coef is defined for t >= 1, but received t={t}.")
        if t > self.config.T:
             print(f"Warning: Calculating loss_coef for t={t} which is > config.T={self.config.T}.")
             # We will allow it for formula consistency, but it might indicate incorrect usage.


        eta_t_val = self.get_eta_t(t)         # Uses cached result
        eta_t_minus_1_val = self.get_eta_t(t - 1) # Uses cached result (get_eta_t handles t-1=0 case)

        # Denominator involves eta_t and eta_{t-1}. Both should be positive
        # based on the formula and eta_1 > 0.
        # Add a small tolerance check for robustness against potential floating point issues.
        denominator = 2 * (self.config.kappa ** 2) * eta_t_val * eta_t_minus_1_val
        if denominator.abs() < 1e-12: # Check if near zero
             # This indicates a potential issue with the formula or parameters leading to zero division.
             raise ValueError(f"Loss coefficient denominator is near zero at t={t}. Denominator: {denominator.item()}")

        alpha_t_val = self.get_alpha_t(t) # Uses cached result (requires t>=1)

        return alpha_t_val / denominator



    def get_noisy_image(
        self,
        t: int,
        image_traget: torch.Tensor,
        image_original: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns the noisy image at time step t according to a specific formula.

        Formula: noisy_image = (1 - eta_t) * image_original + eta_t * image_target + kappa * sqrt(eta_t) * noise

        Args:
            t: The time step (integer). Must be in the range [0, config.T].
            image_traget: The target image tensor.
            image_original: The original image tensor. Should have the same shape,
                            dtype, and device as image_target.

        Returns:
            The noisy image tensor, with the same shape, dtype, and device
            as image_original and image_target.
        """
        if not isinstance(t, int) or t < 0 or t > self.config.T:
            # Limit t to the valid diffusion range [0, T] for noise application
            raise ValueError(f"Time step t must be an integer in the range [0, {self.config.T}], but got {t}.")

        if image_traget.shape != image_original.shape:
             raise ValueError("image_target and image_original must have the same shape.")
        if image_traget.dtype != image_original.dtype:
             print("Warning: image_target and image_original have different dtypes.") # Or raise error?
        if image_traget.device != image_original.device:
             print("Warning: image_target and image_original are on different devices.") # Or raise error?
             # For calculation correctness, move one to the other's device if they aren't already.
             # Let's assume image_original's device is the target device.
             if image_traget.device != image_original.device:
                  image_traget = image_traget.to(image_original.device)
                  print(f"Moved image_target to device {image_original.device}.")


        eta_t_val = self.get_eta_t(t) # Uses cached result

        # Ensure eta_t is a tensor scalar before operations
        if not isinstance(eta_t_val, torch.Tensor) or eta_t_val.numel() != 1:
             raise TypeError(f"get_eta_t returned unexpected type or shape: {type(eta_t_val)}, {eta_t_val.shape if isinstance(eta_t_val, torch.Tensor) else 'N/A'}")

        # Ensure eta_t_val is non-negative for sqrt. Should be guaranteed by formula structure (eta_1 > 0, b_0 > 0)
        if eta_t_val < 0:
             # This would indicate a mathematical issue in get_eta_t or its inputs
             raise ValueError(f"Calculated eta_t is negative at t={t}: {eta_t_val.item()}")


        # Generate noise with the same shape, dtype, and device as the images
        noise = torch.randn_like(image_original)

        # Calculate scheduled noise term
        scheduled_noise_term = self.config.kappa * torch.sqrt(eta_t_val) * noise

        # Calculate scheduled residual term (rewritten form)
        # This is (1 - eta_t) * image_original + eta_t * image_target
        scheduled_residual_term = eta_t_val * image_original + (1.0 - eta_t_val) * image_traget
        # scheduled_residual_term = (1.0 - eta_t_val) * image_original + eta_t_val * image_traget

        # Combine terms
        noisy_image = scheduled_residual_term + scheduled_noise_term

        return noisy_image

# # Example Usage
# if __name__ == "__main__":
#     print("--- Diffusion Scheduler Example ---")
#     try:
#         # Initialize configuration
#         config = CFG(kappa=2.0, p=0.5, eta_T=0.999, T=50)
#         print(f"\nInitialized config: {config}")
#         print(f"Calculated eta_1: {config.eta_1:.6f}")

#         # Initialize the scheduler with the configuration
#         scheduler = DiffusionScheduler(config)
#         print("\nInitialized Diffusion Scheduler.")

#         # Calculate and print parameters for various time steps using the scheduler instance
#         print("\n--- Parameter Values ---")
#         test_times = [0, 1, 5, config.T // 2, config.T]
#         for t in test_times:
#             try:
#                 beta_t = scheduler.get_beta_t(t)
#                 eta_t = scheduler.get_eta_t(t)
#                 print(f"t={t}:")
#                 print(f"  beta_t: {beta_t.item():.6f}")
#                 print(f"  eta_t: {eta_t.item():.6f}")

#                 if t >= 1:
#                     alpha_t = scheduler.get_alpha_t(t)
#                     loss_coef = scheduler.get_loss_coef(t)
#                     print(f"  alpha_t: {alpha_t.item():.6f}")
#                     print(f"  loss_coef_t: {loss_coef.item():.6f}")

#             except ValueError as e:
#                 print(f"t={t}: Error calculating parameters - {e}")

#         # Calculate b_0 once using the scheduler instance
#         try:
#             b_0 = scheduler._get_b_0() # Note: Using protected name _get_b_0 as it's internal
#             print(f"\nCalculated b_0: {b_0.item():.6f}")
#         except ValueError as e:
#             print(f"\nError calculating b_0: {e}")


#         # Example of generating a noisy image
#         print("\n--- Noisy Image Generation Example ---")
#         # Use dummy tensors, potentially on GPU if available
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print(f"Using device: {device}")

#         dummy_original = torch.randn(1, 3, 32, 32, device=device) # Batch, Channel, H, W
#         dummy_target = torch.randn(1, 3, 32, 32, device=device)

#         time_step_for_noise = config.T // 3
#         print(f"Generating noisy image at t={time_step_for_noise}...")

#         try:
#             # Call the method on the scheduler instance
#             noisy_image = scheduler.get_noisy_image(
#                 time_step_for_noise,
#                 dummy_target,
#                 dummy_original,
#             )
#             print(f"Generated noisy image tensor:")
#             print(f"  Shape: {noisy_image.shape}")
#             print(f"  Dtype: {noisy_image.dtype}")
#             print(f"  Device: {noisy_image.device}")
#             # Optional: Print a small part of the tensor to see values
#             # print(noisy_image[0, 0, :2, :2])

#         except ValueError as e:
#             print(f"Error generating noisy image: {e}")


#         # Test validation in get_noisy_image
#         print("\n--- Testing get_noisy_image validation ---")
#         try:
#              scheduler.get_noisy_image(config.T + 1, dummy_target, dummy_original)
#         except ValueError as e:
#              print(f"Caught expected error for t > T: {e}")

#         try:
#              scheduler.get_noisy_image(-1, dummy_target, dummy_original)
#         except ValueError as e:
#              print(f"Caught expected error for t < 0: {e}")

#         try:
#              dummy_wrong_shape = torch.randn(1, 3, 16, 16, device=device)
#              scheduler.get_noisy_image(time_step_for_noise, dummy_target, dummy_wrong_shape)
#         except ValueError as e:
#              print(f"Caught expected error for shape mismatch: {e}")


#         # Print cache info to see if caching worked
#         print("\n--- Cache Info ---")
#         print(f"_get_b_0 cache info: {scheduler._get_b_0.cache_info()}")
#         print(f"get_beta_t cache info: {scheduler.get_beta_t.cache_info()}")
#         print(f"get_eta_t cache info: {scheduler.get_eta_t.cache_info()}")
#         print(f"get_alpha_t cache info: {scheduler.get_alpha_t.cache_info()}")
#         print(f"get_loss_coef cache info: {scheduler.get_loss_coef.cache_info()}")

#         # Create another scheduler with different config to show caches are separate
#         print("\n--- Another Scheduler Instance ---")
#         config_2 = CFG(kappa=3.0, p=0.8, eta_T=0.99, T=100)
#         scheduler_2 = DiffusionScheduler(config_2)
#         print(f"Initialized second config: {config_2}")
#         # Access a parameter to trigger calculation and caching in the second instance
#         scheduler_2.get_eta_t(config_2.T // 2)
#         print(f"Second scheduler get_eta_t cache info: {scheduler_2.get_eta_t.cache_info()}")
#         # First scheduler cache info should be unchanged
#         print(f"First scheduler get_eta_t cache info: {scheduler.get_eta_t.cache_info()}")


#     except ValueError as e:
#         print(f"\nError during setup or execution: {e}")
#     except TypeError as e:
#          print(f"\nError during setup or execution: {e}")