import numpy as np
import torch

from rlbidder.constants import HISTORY_FEATURE_DIM, NUM_TICKS, STATE_DIM
from rlbidder.evaluation.history import StepwiseAuctionHistory


def masked_mean(
    mask: torch.Tensor,
    value: torch.Tensor,
    dim: int | None = None,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Compute the mean of the value tensor along the specified dimensions,
    ignoring the dimensions where the mask is False.
    
    Args:
        mask: A boolean tensor of shape (batch_size, seq_len) or (batch_size, seq_len, ...)
        value: A tensor of shape (batch_size, seq_len, ...)
        dim: The dimensions to reduce. If None, all dimensions are reduced.
        eps: A small constant to avoid division by zero.
        
    Returns:
        A tensor of shape (batch_size, ...) with the mean of the value tensor along the specified dimensions,
        ignoring the dimensions where the mask is False.
    """
    mask = mask.expand(*value.shape)
    return torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))


def grad_total_norm(
    loss: torch.Tensor,
    params: list[torch.nn.Parameter] | tuple[torch.nn.Parameter, ...],
    p: float = 2.0,
) -> torch.Tensor:
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    grads = [g.detach() for g in grads if g is not None]
    return torch.nn.utils.get_total_norm(grads, norm_type=p, foreach=True)


def extract_state_features(
    adv_indices: np.ndarray,
    timeStepIndex: int,
    budget: np.ndarray,
    remaining_budget: np.ndarray,
    auction_history: StepwiseAuctionHistory,
    pValues: np.ndarray,
    total_ticks: int = NUM_TICKS,
) -> np.ndarray:
    """Extract state features for bidding agents.
    
    Args:
        adv_indices: Indices of advertisers handled by this agent
        timeStepIndex: Current time step index
        budget: Initial budget for each advertiser
        remaining_budget: Remaining budget for each advertiser
        auction_history: Historical auction data
        pValues: P-values matrix with shape (num_pv, num_advertisers)
        total_ticks: Total number of time steps (default: 48)
        
    Returns:
        np.ndarray: State features matrix with shape (num_advertisers, STATE_DIM)
    """
    # Sanity checks for input shapes and values
    if not isinstance(adv_indices, np.ndarray) or adv_indices.ndim != 1:
        raise ValueError("adv_indices must be a 1D numpy array")
    
    if not isinstance(budget, np.ndarray) or budget.shape != adv_indices.shape:
        raise ValueError(f"budget must be a 1D numpy array with same shape as adv_indices, got {budget.shape}")
    
    if not isinstance(remaining_budget, np.ndarray) or remaining_budget.shape != adv_indices.shape:
        raise ValueError("remaining_budget must be a 1D numpy array with same shape as adv_indices")
    
    if not isinstance(pValues, np.ndarray) or pValues.shape[1] != len(adv_indices):
        raise ValueError("pValues must be a 2D numpy array with same number of columns as adv_indices length")

    if total_ticks <= 0:
        raise ValueError("total_ticks must be positive")
    
    if timeStepIndex < 0 or timeStepIndex >= total_ticks:
        raise ValueError(f"timeStepIndex must be in range [0, {total_ticks})")
    
    # states has shape (num_advertisers, STATE_DIM)
    states = np.zeros((len(adv_indices), STATE_DIM), dtype=np.float32)
    
    # Compute normalized time remaining
    time_left = float(total_ticks - timeStepIndex) / total_ticks
    # Compute budget_left as a vector for all advertisers handled by this agent
    budget_left = np.divide(
        remaining_budget, 
        budget, 
        out=np.zeros_like(remaining_budget, dtype=float), 
        where=(budget > 0),
    )
    # Prepend time_left and budget_left as first two columns
    states[:, 0] = time_left
    states[:, 1] = budget_left
    
    # Prepare history features as (num_advertisers, 13)
    history_features = auction_history.get_state_features()[adv_indices]
    if history_features.shape != (len(adv_indices), HISTORY_FEATURE_DIM):
        raise ValueError(
            f"history_features must have shape ({len(adv_indices)}, {HISTORY_FEATURE_DIM}), {history_features.shape=}"
        )
    
    # Fill in first 10 features from history_features to states from state[:, 2]
    # - from `avg_bid_all` to `avg_xi_last_3`
    states[:, 2:12] = history_features[:, :10]

    # Fill in avg_pvalues and timeStepIndex_volume as features 12 and 13
    # pValues has shape (num pv, num_advertisers)
    avg_pvalues = pValues.mean(axis=0)
    timeStepIndex_volume = pValues.shape[0]
    states[:, 12] = avg_pvalues
    states[:, 13] = timeStepIndex_volume
    
    # Fill in the remaining features from history_features into states
    # history_features has shape (num_advertisers, 12)
    # We already used columns :10 for states[:, 2:12], so fill columns 10 and 11 into states[:, 14:16]
    states[:, 14:16] = history_features[:, 10:12]

    # Fill in the CPA compliance ratio
    states[:, 16] = history_features[:, 12]

    return states
