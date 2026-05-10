# Copyright 2025 The Torch-Spyre Authors.
# SPDX-License-Identifier: Apache-2.0
"""Utilities for loading SEN memory-map dumps into NumPy arrays and
seeding an ``HBMSimulator`` from them.

Intended for tests under ``tests/sdsc/`` that want to run ktir-cpu
against the exact inputs the SDSC reference pipeline produced, and
compare against the exact expected outputs.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from ..dtypes import bytes_per_elem, to_np_dtype

# SEN HBM sticks are 128 bytes.  Matches HBMSimulator.STICK_BYTES.
_STICK = 128


@dataclass(frozen=True)
class TensorDescriptor:
    """Where a tensor lives in HBM and how to shape it.

    Attributes:
        element_addr: Address as emitted in the translator's MLIR
            (element offset, not bytes).
        byte_addr: Corresponding byte address in HBM.  Must equal
            ``element_addr * bytes_per_elem(dtype)``; enforced in
            ``__post_init__``.  Must land on a stick boundary.
        shape: Tensor shape as a tuple.
        dtype: KTIR dtype string ("f16", "f32", "i32", ...).
    """

    element_addr: int
    byte_addr: int
    shape: tuple
    dtype: str

    def __post_init__(self):
        expected = self.element_addr * bytes_per_elem(self.dtype)
        if self.byte_addr != expected:
            raise ValueError(
                f"byte_addr={self.byte_addr} != element_addr({self.element_addr})"
                f" * itemsize({bytes_per_elem(self.dtype)}) = {expected}"
            )
        if self.byte_addr % _STICK != 0:
            raise ValueError(
                f"byte_addr={self.byte_addr} is not aligned to "
                f"{_STICK}-byte sticks"
            )

# SEN datatype labels in the map header map to ktir dtype strings.
_DATATYPE_TO_KTIR = {
    "SEN169_FP16": "f16",
}


def _parse_map(path: Path) -> Tuple[str, Dict[int, np.ndarray]]:
    """Return (ktir_dtype, {stick_index: 64-element f16 array}).

    The decimal format is one row per stick:
        <decimal_stick_index>: <64 space-separated float values>

    Sections separated by repeated '# datatype: ...' headers in the body
    are allowed — comment lines are skipped. Stick indices must not
    repeat across sections.
    """
    sticks: Dict[int, np.ndarray] = {}
    ktir_dtype: str | None = None
    with open(path) as f:
        header = f.readline().rstrip("\n")
        m = re.match(r"#\s*datatype:\s*(\S+)\s*$", header)
        if not m:
            raise ValueError(
                f"{path}: expected '# datatype: <NAME>' on line 1, got {header!r}"
            )
        raw_dtype = m.group(1)
        if raw_dtype not in _DATATYPE_TO_KTIR:
            raise ValueError(
                f"{path}: unsupported datatype {raw_dtype!r}; "
                f"supported: {sorted(_DATATYPE_TO_KTIR)}"
            )
        ktir_dtype = _DATATYPE_TO_KTIR[raw_dtype]
        for lineno, line in enumerate(f, start=2):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                raise ValueError(
                    f"{path}:{lineno}: expected '<stick_index>: <values...>', "
                    f"missing ':'"
                )
            idx_str, rest = line.split(":", 1)
            try:
                stick = int(idx_str.strip())
            except ValueError as e:
                raise ValueError(
                    f"{path}:{lineno}: invalid stick index {idx_str.strip()!r}"
                ) from e
            tokens = rest.split()
            if len(tokens) != 64:
                raise ValueError(
                    f"{path}:{lineno}: expected 64 values per stick, got {len(tokens)}"
                )
            try:
                values = np.array([float(t) for t in tokens], dtype=np.float16)
            except ValueError as e:
                raise ValueError(
                    f"{path}:{lineno}: could not parse floats"
                ) from e
            if stick in sticks:
                raise ValueError(
                    f"{path}:{lineno}: duplicate stick index {stick} "
                    f"(first seen on an earlier row)"
                )
            sticks[stick] = values
    return ktir_dtype, sticks


def load_memory_map(
    path, descriptors: Dict[str, TensorDescriptor],
) -> Dict[str, np.ndarray]:
    """Parse a SEN memory map and slice out each named tensor.

    Args:
        path: Path to the map file.
        descriptors: Dict of name -> TensorDescriptor.

    Returns:
        Dict of name -> np.ndarray with the descriptor's shape and dtype.

    Raises:
        ValueError: on malformed file, unsupported datatype, dtype
            mismatch between descriptor and file, descriptor size not a
            multiple of stick size, range out of map bounds, or hole in
            the descriptor's stick range.
    """
    path = Path(path)
    file_dtype, sticks = _parse_map(path)
    if not sticks:
        raise ValueError(f"{path}: no data rows")
    min_byte = min(sticks) * _STICK
    max_byte = (max(sticks) + 1) * _STICK  # exclusive

    out: Dict[str, np.ndarray] = {}
    for name, desc in descriptors.items():
        if desc.dtype != file_dtype:
            raise ValueError(
                f"descriptor {name!r} dtype {desc.dtype!r} != "
                f"map dtype {file_dtype!r}"
            )
        nbytes = int(np.prod(desc.shape)) * bytes_per_elem(desc.dtype)
        if nbytes % _STICK != 0:
            raise ValueError(
                f"descriptor {name!r} size={nbytes} bytes is not a multiple "
                f"of stick size ({_STICK})"
            )
        end_byte = desc.byte_addr + nbytes
        if desc.byte_addr < min_byte or end_byte > max_byte:
            raise ValueError(
                f"descriptor {name!r} spans bytes [{desc.byte_addr}, {end_byte}); "
                f"map covers [{min_byte}, {max_byte})"
            )
        pieces = []
        for byte_off in range(desc.byte_addr, end_byte, _STICK):
            stick_idx = byte_off // _STICK
            if stick_idx not in sticks:
                raise ValueError(
                    f"descriptor {name!r} has a missing stick at byte {byte_off}"
                )
            pieces.append(sticks[stick_idx])
        flat = np.concatenate(pieces).astype(to_np_dtype(desc.dtype), copy=False)
        out[name] = flat.reshape(desc.shape).copy()
    return out


def write_tensors_to_hbm(hbm, tensors, descriptors):
    """Seed HBM with each tensor at its descriptor's byte_addr.

    Args:
        hbm: An HBMSimulator instance.
        tensors: Dict of name -> np.ndarray.
        descriptors: Dict of name -> TensorDescriptor. Must have the
            same keys as ``tensors``.

    Raises:
        ValueError: on key set mismatch, shape mismatch, or dtype mismatch.
    """
    if set(tensors) != set(descriptors):
        raise ValueError(
            f"tensors keys {sorted(tensors)} != descriptors keys "
            f"{sorted(descriptors)}"
        )
    for name, arr in tensors.items():
        desc = descriptors[name]
        if tuple(arr.shape) != tuple(desc.shape):
            raise ValueError(
                f"tensor {name!r}: shape {tuple(arr.shape)} != "
                f"descriptor shape {tuple(desc.shape)}"
            )
        if np.dtype(arr.dtype) != to_np_dtype(desc.dtype):
            raise ValueError(
                f"tensor {name!r}: dtype {arr.dtype} != "
                f"descriptor dtype {desc.dtype}"
            )
        hbm.write(desc.byte_addr, arr)
