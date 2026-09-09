"""2D interface to the shared lazy bit-packed cloud dataset adapter."""

from runtime.cloud_bitpacked_3d import (
    dataset_from_bitpacked_trajectories as _dataset_from_bitpacked,
)


def dataset_from_bitpacked_trajectories(*args, **kwargs):
    """Build six-feature inputs and transition-aware 2D targets lazily."""
    kwargs["spatial_dims"] = 2
    return _dataset_from_bitpacked(*args, **kwargs)
