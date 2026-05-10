# Copyright 2025 The Torch-Spyre Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ktir_cpu.testing.memory_maps."""

from pathlib import Path

import numpy as np
import pytest

from ktir_cpu.memory import HBMSimulator
from ktir_cpu.testing.memory_maps import TensorDescriptor, load_memory_map, write_tensors_to_hbm


def test_public_reexports_from_subpackage():
    """Importing from ktir_cpu.testing (not .memory_maps) must work."""
    from ktir_cpu.testing import (
        TensorDescriptor,
        load_memory_map,
        write_tensors_to_hbm,
    )
    # sanity: same objects as submodule
    from ktir_cpu.testing import memory_maps as mm
    assert TensorDescriptor is mm.TensorDescriptor
    assert load_memory_map is mm.load_memory_map
    assert write_tensors_to_hbm is mm.write_tensors_to_hbm


class TestTensorDescriptor:
    def test_valid_f16(self):
        d = TensorDescriptor(
            element_addr=64000, byte_addr=128000,
            shape=(12, 64, 64), dtype="f16",
        )
        assert d.element_addr == 64000
        assert d.byte_addr == 128000
        assert d.shape == (12, 64, 64)
        assert d.dtype == "f16"

    def test_rejects_byte_addr_not_matching_element_times_itemsize(self):
        with pytest.raises(ValueError, match="byte_addr"):
            TensorDescriptor(
                element_addr=64000, byte_addr=64000,   # should be 128000 for f16
                shape=(12, 64, 64), dtype="f16",
            )

    def test_rejects_unaligned_byte_addr(self):
        # 128008 is a multiple of 2 (consistent with element_addr=64004 f16)
        # but not a multiple of the 128-byte stick size.
        with pytest.raises(ValueError, match="aligned"):
            TensorDescriptor(
                element_addr=64004, byte_addr=128008,
                shape=(4,), dtype="f16",
            )


def _write_map(tmp_path, datatype, rows):
    """rows = list of (stick_index, np.ndarray[64] or list of 64 floats).

    Writes the new decimal memory-map format.
    """
    path = tmp_path / "map.txt"
    lines = [f"# datatype: {datatype}"]
    for stick_idx, values in rows:
        arr = np.asarray(values, dtype=np.float16)
        assert arr.shape == (64,), f"expected 64 values, got shape {arr.shape}"
        # Use repr-enough precision for f16
        tokens = " ".join(repr(float(v)) for v in arr)
        lines.append(f"{stick_idx}: {tokens}")
    path.write_text("\n".join(lines) + "\n")
    return path


class TestLoadMemoryMapHappyPath:
    def test_single_stick_f16(self, tmp_path):
        # 64 f16 values = one full 128-byte stick.
        data = np.arange(64, dtype=np.float16)
        path = _write_map(tmp_path, "SEN169_FP16", [(1000, data)])

        desc = TensorDescriptor(
            element_addr=64000, byte_addr=128000,
            shape=(64,), dtype="f16",
        )
        out = load_memory_map(path, {"A": desc})

        assert set(out) == {"A"}
        assert out["A"].dtype == np.float16
        assert out["A"].shape == (64,)
        assert np.array_equal(out["A"], data)

    def test_multi_stick_contiguous(self, tmp_path):
        # 256 f16 = 4 sticks
        data = np.arange(256, dtype=np.float16)
        sticks = [
            (500 + i, data[i * 64 : (i + 1) * 64])
            for i in range(4)
        ]
        path = _write_map(tmp_path, "SEN169_FP16", sticks)

        desc = TensorDescriptor(
            element_addr=32000, byte_addr=64000,
            shape=(8, 32), dtype="f16",
        )
        out = load_memory_map(path, {"T": desc})
        assert out["T"].shape == (8, 32)
        assert np.array_equal(out["T"].reshape(-1), data)

    def test_multiple_descriptors_from_one_map(self, tmp_path):
        a_data = np.arange(64, dtype=np.float16)
        b_data = (np.arange(64, dtype=np.float16) + 1000.0).astype(np.float16)
        path = _write_map(tmp_path, "SEN169_FP16", [
            (1000, a_data),
            (1001, b_data),
        ])

        descs = {
            "A": TensorDescriptor(element_addr=64000, byte_addr=128000,
                                  shape=(64,), dtype="f16"),
            "B": TensorDescriptor(element_addr=64064, byte_addr=128128,
                                  shape=(64,), dtype="f16"),
        }
        out = load_memory_map(path, descs)
        assert np.array_equal(out["A"], a_data)
        assert np.array_equal(out["B"], b_data)

    def test_returned_arrays_are_writable(self, tmp_path):
        # Loader must copy so callers can mutate without surprise.
        data = np.zeros(64, dtype=np.float16)
        path = _write_map(tmp_path, "SEN169_FP16", [(1000, data)])
        desc = TensorDescriptor(
            element_addr=64000, byte_addr=128000,
            shape=(64,), dtype="f16",
        )
        out = load_memory_map(path, {"A": desc})
        out["A"][0] = np.float16(42.0)  # must not raise
        assert float(out["A"][0]) == 42.0


class TestLoadMemoryMapErrors:
    def test_missing_header_raises(self, tmp_path):
        path = tmp_path / "bad.txt"
        path.write_text("not a header\n")
        with pytest.raises(ValueError, match="datatype"):
            load_memory_map(path, {})

    def test_wrong_header_format_raises(self, tmp_path):
        path = tmp_path / "bad.txt"
        path.write_text("# not-a-datatype-line\n")
        with pytest.raises(ValueError, match="datatype"):
            load_memory_map(path, {})

    def test_unsupported_datatype_raises(self, tmp_path):
        path = tmp_path / "bad.txt"
        path.write_text("# datatype: SEN169_FP32\n")
        with pytest.raises(ValueError, match="unsupported datatype"):
            load_memory_map(path, {})

    def test_bad_row_format_missing_colon_raises(self, tmp_path):
        path = tmp_path / "bad.txt"
        path.write_text("# datatype: SEN169_FP16\n1000 0.1 0.2\n")
        with pytest.raises(ValueError, match="missing ':'"):
            load_memory_map(path, {})

    def test_wrong_value_count_raises(self, tmp_path):
        path = tmp_path / "bad.txt"
        short_row = "1000: " + " ".join("0.0" for _ in range(32))
        path.write_text(f"# datatype: SEN169_FP16\n{short_row}\n")
        with pytest.raises(ValueError, match="expected 64 values"):
            load_memory_map(path, {})

    def test_invalid_stick_index_raises(self, tmp_path):
        path = tmp_path / "bad.txt"
        row = "abc: " + " ".join("0.0" for _ in range(64))
        path.write_text(f"# datatype: SEN169_FP16\n{row}\n")
        with pytest.raises(ValueError, match="invalid stick index"):
            load_memory_map(path, {})

    def test_invalid_float_value_raises(self, tmp_path):
        path = tmp_path / "bad.txt"
        row = "1000: " + "foo " + " ".join("0.0" for _ in range(63))
        path.write_text(f"# datatype: SEN169_FP16\n{row}\n")
        with pytest.raises(ValueError, match="could not parse floats"):
            load_memory_map(path, {})

    def test_empty_map_raises(self, tmp_path):
        path = tmp_path / "bad.txt"
        path.write_text("# datatype: SEN169_FP16\n")
        with pytest.raises(ValueError, match="no data rows"):
            load_memory_map(path, {})

    def test_tensor_size_not_multiple_of_stick_raises(self, tmp_path):
        data = np.arange(64, dtype=np.float16)
        path = _write_map(tmp_path, "SEN169_FP16", [(1000, data)])
        desc = TensorDescriptor(
            element_addr=64000, byte_addr=128000,
            shape=(4,), dtype="f16",
        )
        with pytest.raises(ValueError, match="multiple of stick size"):
            load_memory_map(path, {"A": desc})

    def test_range_out_of_bounds_raises(self, tmp_path):
        data = np.arange(64, dtype=np.float16)
        path = _write_map(tmp_path, "SEN169_FP16", [(1000, data)])
        desc = TensorDescriptor(
            element_addr=64000, byte_addr=128000,
            shape=(128,), dtype="f16",  # 256 bytes = 2 sticks
        )
        with pytest.raises(ValueError, match="map covers"):
            load_memory_map(path, {"A": desc})

    def test_hole_in_range_raises(self, tmp_path):
        data = np.arange(64, dtype=np.float16)
        path = _write_map(tmp_path, "SEN169_FP16", [
            (1000, data), (1002, data),
        ])
        desc = TensorDescriptor(
            element_addr=64000, byte_addr=128000,
            shape=(192,), dtype="f16",  # 384 bytes = 3 sticks
        )
        with pytest.raises(ValueError, match="missing stick"):
            load_memory_map(path, {"A": desc})

    def test_dtype_mismatch_raises(self, tmp_path):
        data = np.arange(64, dtype=np.float16)
        path = _write_map(tmp_path, "SEN169_FP16", [(1000, data)])
        desc = TensorDescriptor(
            element_addr=32000, byte_addr=128000,
            shape=(32,), dtype="f32",
        )
        with pytest.raises(ValueError, match="dtype"):
            load_memory_map(path, {"A": desc})

    def test_duplicate_stick_index_raises(self, tmp_path):
        data = np.arange(64, dtype=np.float16)
        path = _write_map(tmp_path, "SEN169_FP16", [
            (1000, data), (1000, data),
        ])
        with pytest.raises(ValueError, match="duplicate stick index"):
            load_memory_map(path, {})


class TestWriteTensorsToHbm:
    def test_writes_arrays_at_correct_addresses(self):
        hbm = HBMSimulator()
        a = np.arange(64, dtype=np.float16)
        b = (np.arange(64, dtype=np.float16) * 2).astype(np.float16)

        descs = {
            "A": TensorDescriptor(element_addr=64000, byte_addr=128000,
                                  shape=(64,), dtype="f16"),
            "B": TensorDescriptor(element_addr=64064, byte_addr=128128,
                                  shape=(64,), dtype="f16"),
        }
        write_tensors_to_hbm(hbm, {"A": a, "B": b}, descs)

        got_a = hbm.read(128000, 64, "f16")
        got_b = hbm.read(128128, 64, "f16")
        assert np.array_equal(got_a, a)
        assert np.array_equal(got_b, b)

    def test_key_mismatch_raises(self):
        hbm = HBMSimulator()
        descs = {
            "A": TensorDescriptor(element_addr=64000, byte_addr=128000,
                                  shape=(64,), dtype="f16"),
        }
        with pytest.raises(ValueError, match="keys"):
            write_tensors_to_hbm(
                hbm, {"B": np.zeros(64, dtype=np.float16)}, descs,
            )

    def test_shape_mismatch_raises(self):
        hbm = HBMSimulator()
        descs = {
            "A": TensorDescriptor(element_addr=64000, byte_addr=128000,
                                  shape=(64,), dtype="f16"),
        }
        with pytest.raises(ValueError, match="shape"):
            write_tensors_to_hbm(
                hbm, {"A": np.zeros(32, dtype=np.float16)}, descs,
            )

    def test_dtype_mismatch_raises(self):
        hbm = HBMSimulator()
        descs = {
            "A": TensorDescriptor(element_addr=64000, byte_addr=128000,
                                  shape=(64,), dtype="f16"),
        }
        with pytest.raises(ValueError, match="dtype"):
            write_tensors_to_hbm(
                hbm, {"A": np.zeros(64, dtype=np.float32)}, descs,
            )
