"""Testing helpers for ktir-cpu (memory-map loading, etc.)."""

from .memory_maps import (
    TensorDescriptor,
    load_memory_map,
    write_tensors_to_hbm,
)

__all__ = [
    "TensorDescriptor",
    "load_memory_map",
    "write_tensors_to_hbm",
]
