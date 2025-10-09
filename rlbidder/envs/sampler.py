import hashlib
import os

import numpy as np
import torch
from scipy.stats import truncnorm

from rlbidder.constants import DEFAULT_SEED


class ValueSampler:
    def __init__(
        self,
        num_advertisers: int,
        seed: int = DEFAULT_SEED,
        lower_bound_std: float = -2,
        upper_bound_std: float = 2,
    ) -> None:
        # Set thread limits to prevent oversubscription in multiprocessing
        os.environ.update({
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", 
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1"
        })
        
        self.num_advertisers = num_advertisers

        self.rng = np.random.Generator(np.random.PCG64(seed))
        # Create a deterministic seed using hashlib and the key "{seed}-{num_advertisers}"
        hash_key = f"{seed}-{num_advertisers}".encode()
        hash_digest = hashlib.sha256(hash_key).digest()
        seed_int = int.from_bytes(hash_digest[:8], "big")
        self.rng_bounds = np.random.Generator(np.random.PCG64(seed_int))

        if lower_bound_std >= 0:
            raise ValueError(f"lower_bound_std should be negative (got {lower_bound_std}).")
        if upper_bound_std <= 0:
            raise ValueError(f"upper_bound_std should be positive (got {upper_bound_std}).")
        
        # Sample different bounds for each advertiser
        # sample lower bounds: U[0, 1] * lower_bound_std
        # sample upper bounds: U[0, 1] * upper_bound_std
        self.lower_bounds = self.rng_bounds.uniform(0.1, 1, (1, num_advertisers)) * lower_bound_std
        self.upper_bounds = self.rng_bounds.uniform(0.1, 1, (1, num_advertisers)) * upper_bound_std

    def sample(
        self,
        pValues: np.ndarray,
        pValueSigmas: np.ndarray,
        sample_conversions: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        # lower_bound, upper_bound = self._sample_bounds()
        # Convert lower and upper bounds from standard normal coordinates to truncnorm's required a, b
        # a = (lower_bound - mu) / sigma, b = (upper_bound - mu) / sigma
        # Handle zeros in p_value_sigmas: set a and b to zero where sigma is zero
        # Use numerical tolerance to check for zeros in p_value_sigmas
        # mask = np.isclose(pValueSigmas, 0, atol=1e-8, rtol=1e-8)
        # The a and b that you pass to truncnorm are not the raw truncation points on the x-axis.
        # Instead:
        # They are expressed in standard deviations away from the mean of the untruncated normal.
        
        values = (
            truncnorm.rvs(
                a=self.lower_bounds,
                b=self.upper_bounds,
                loc=pValues,
                scale=pValueSigmas,
                random_state=self.rng,
            )
            .clip(0, 1)
        )

        if sample_conversions:
            conversions = self.rng.binomial(n=1, p=values)
            return values, conversions
        else:
            return values


def sample_pValues_and_conversions_scipy(
    pValues: list[np.ndarray],
    pValueSigmas: list[np.ndarray],
    seed: int = DEFAULT_SEED,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Sample pValues and conversions using scipy's truncnorm via ValueSampler.

    Args:
        pValues: list of np.ndarray, each (n, d)
        pValueSigmas: list of np.ndarray, each (n, d)
        seed: int, random seed

    Returns:
        sampled_pValues_list: list of np.ndarray (same shape as input)
        sampled_conversion_list: list of np.ndarray (same shape as input)
    """
    # Set thread limits to prevent oversubscription in multiprocessing
    os.environ.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", 
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1"
    })
    
    # Get num_advertisers from the shape of the first pValues array
    num_advertisers = pValues[0].shape[1]
    
    # Create ValueSampler
    sampler = ValueSampler(
        num_advertisers=num_advertisers, 
        seed=seed, 
        lower_bound_std=-2, 
        upper_bound_std=2,
    )
    
    # Concatenate all ticks for batch sampling
    batch_pValues = np.concatenate(pValues, axis=0)
    batch_pValueSigmas = np.concatenate(pValueSigmas, axis=0)

    # Sample pValues and conversions using ValueSampler
    sampled_pValues, sampled_conversions = sampler.sample(batch_pValues, batch_pValueSigmas, sample_conversions=True)
    
    # Split back to list of numpy arrays per tick
    tick_split_indices = np.cumsum([p.shape[0] for p in pValues])[:-1]
    sampled_pValues_list = np.split(sampled_pValues, tick_split_indices, axis=0)
    sampled_conversion_list = np.split(sampled_conversions, tick_split_indices, axis=0)
    return sampled_pValues_list, sampled_conversion_list


@torch.inference_mode()
def sample_pValues_and_conversions_torch(
    pValues: list[np.ndarray],
    pValueSigmas: list[np.ndarray],
    seed: int = DEFAULT_SEED,
    device: str = "cpu",
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Fastest version: uses torch for vectorized sampling.

    Args:
        pValues, pValueSigmas: list of np.ndarray, each (n, d)
        seed: int, random seed
        device: "cpu" or "cuda"

    Returns:
        sampled_pValues_list: list of torch.Tensor (same shape as input)
        sampled_conversion_list: list of torch.Tensor (same shape as input)

    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    # Concatenate all ticks for batch sampling
    batch_pValues = torch.from_numpy(np.concatenate(pValues, axis=0)).to(device)
    batch_pValueSigmas = torch.from_numpy(np.concatenate(pValueSigmas, axis=0)).to(device)

    # Sample normal and clip to [0, 1] using the provided generator
    sampled_pValues = torch.normal(batch_pValues, batch_pValueSigmas, generator=g).clip(0, 1)
    # Sample binomial (Bernoulli) for conversions using the generator
    sampled_conversion = torch.bernoulli(sampled_pValues, generator=g).to(torch.int32)

    # Split back to list of tensors per tick
    tick_sizes = [p.shape[0] for p in pValues]
    sampled_pValues_list = torch.split(sampled_pValues, tick_sizes, dim=0)
    sampled_conversion_list = torch.split(sampled_conversion, tick_sizes, dim=0)
    # Convert to list of numpy arrays
    sampled_pValues_list = [p.detach().cpu().numpy() for p in sampled_pValues_list]
    sampled_conversion_list = [c.detach().cpu().numpy() for c in sampled_conversion_list]
    return sampled_pValues_list, sampled_conversion_list
