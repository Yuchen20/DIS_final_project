import torch
import functools
from typing import Union

@functools.lru_cache(maxsize=1000)
def get_b_0(config) -> torch.Tensor:
    """Calculate b_0 parameter for noise scheduling.
    
    Args:
        config: Configuration object containing T, eta_T and eta_1 parameters
    
    Returns:
        torch.Tensor: The calculated b_0 value
    """
    return torch.exp(
        1 / (2 * config.T - 2) * torch.log(
            torch.tensor(config.eta_T / config.eta_1)
        )
    )

@functools.lru_cache(maxsize=1000)
def get_beta_t(t: torch.Tensor, config) -> torch.Tensor:
    """Calculate beta_t parameter for noise scheduling.
    
    Args:
        t: Time step (torch.Tensor)
        config: Configuration object containing T and p parameters
    
    Returns:
        torch.Tensor: The calculated beta_t value
    """

    return torch.pow((t - 1) / (config.T - 1), config.p) * (config.T - 1)


@functools.lru_cache(maxsize=1000)
def get_eta_t(t: torch.Tensor, config) -> torch.Tensor:
    """Calculate eta_t parameter for noise scheduling.
    
    Args:
        t: Time step (torch.Tensor)
        config: Configuration object containing necessary parameters
    
    Returns:
        torch.Tensor: The calculated eta_t value
    """
    return config.eta_1 \
            * torch.pow(get_b_0(config), get_beta_t(t, config) * 2)

@functools.lru_cache(maxsize=1000)
def get_alpha_t(t: torch.Tensor, config) -> torch.Tensor:
    """Calculate alpha_t parameter for noise scheduling.
    
    Args:
        t: Time step (torch.Tensor)
        config: Configuration object containing necessary parameters
    
    Returns:
        torch.Tensor: The calculated alpha_t value
    """
    return get_beta_t(t, config) - get_beta_t(t - 1, config)

@functools.lru_cache(maxsize=1000)
def get_loss_coef(t: torch.Tensor, config) -> torch.Tensor:
    """Calculate loss coefficient for noise scheduling.
    
    Args:
        t: Time step (torch.Tensor)
        config: Configuration object containing kappa and other parameters
    
    Returns:
        torch.Tensor: The calculated loss coefficient
    """
    return get_alpha_t(t, config) / (
        2 * (config.kappa ** 2) * get_eta_t(t, config) * get_eta_t(t - 1, config)
    )

# Sanity checks
def run_sanity_checks(config):
    assert torch.isclose(get_beta_t(torch.tensor(1.), config), 
                        torch.tensor(0.)), "The beta value at t=1 is expected to be 0"
    assert torch.isclose(get_beta_t(torch.tensor(float(config.T)), config), 
                        torch.tensor(float(config.T)), rtol=1e-2), "The beta value at t=T is expected to be T"
    t = torch.randn(64)
    # check output shapes
    assert get_b_0(config).shape == torch.Size([]), "b_0 should be a scalar"
    assert get_beta_t(t, config).shape == t.shape, "beta_t should have the same shape as t"
    assert get_eta_t(t, config).shape == t.shape, "eta_t should have the same shape as t"
    assert get_alpha_t(t, config).shape == t.shape, "alpha_t should have the same shape as t"
    assert get_loss_coef(t, config).shape == t.shape, "loss_coef should have the same shape as t"
    