# Copyright 2025 The Torch-Spyre Authors.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end test covering the newly added linalg named ops via a
ktdp-wrapped MLIR fixture (elementwise add + exp)."""

from pathlib import Path

import numpy as np

from ktir_cpu import KTIRInterpreter


FIXTURE = (
    Path(__file__).parent.parent
    / "examples" / "translator-like" / "elementwise_addexp_ktir.mlir"
)


def test_addexp_kernel_end_to_end():
    """Kernel computes output = exp(x + y) using linalg.add and linalg.exp."""
    interp = KTIRInterpreter()
    interp.load(str(FIXTURE))

    x_ptr, y_ptr, output_ptr, BLOCK_SIZE = interp.arg_names("addexp_kernel")
    sizes = interp.tensor_input_output_sizes("addexp_kernel")
    (n,) = sizes[x_ptr]["shape"]

    rng = np.random.default_rng(42)
    # Keep inputs small so exp doesn't overflow f16 (f16 max ~65504; exp(11) ~60000).
    x = rng.uniform(-1.0, 1.0, size=n).astype(np.float16)
    y = rng.uniform(-1.0, 1.0, size=n).astype(np.float16)
    output = np.zeros(n, dtype=np.float16)

    outputs = interp.execute_function("addexp_kernel", **{
        x_ptr: x, y_ptr: y, output_ptr: output, BLOCK_SIZE: 128,
    })

    result = outputs[output_ptr]
    # Reference computation stays in f16 to mirror the interpreter's
    # precision path (handlers operate on f16 tile data directly).
    expected = np.exp((x + y).astype(np.float16)).astype(np.float16)
    np.testing.assert_allclose(result, expected, rtol=5e-2, atol=5e-2)
