"""Regression tests: the v1.3.1 Colab training crash + GPU-wheel revert.

Covers the two production defects fixed in v1.3.2:

1. torch.compile/Inductor crash (assert_size_stride) on the first training
   step — root causes fixed in commons.slice_segments/slice_segments2
   (vectorised gather) and GeneratorNSF's source feed (dim=1 reshape);
   plus the runtime _CompileGuard that swaps a broken graph to eager.
2. (documentation/notebook guard) — UV_NO_SYNC is asserted in every
   notebook cell that runs `uv run` (a plain `uv run` re-syncs the venv
   from uv.lock and reverts CUDA torch wheels to CPU builds).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from rvc_platform.engine.rvc.models import commons
from rvc_platform.training.trainer import Trainer


# --------------------------------------------------------------------------
# reference (upstream) implementations — the gather must match them exactly
# --------------------------------------------------------------------------
def _ref_slice_segments(x, ids_str, segment_size=4):
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        ret[i] = x[i, :, ids_str[i]:ids_str[i] + segment_size]
    return ret


def _ref_slice_segments2(x, ids_str, segment_size=4):
    ret = torch.zeros_like(x[:, :segment_size])
    for i in range(x.size(0)):
        ret[i] = x[i, ids_str[i]:ids_str[i] + segment_size]
    return ret


class TestSliceSegmentsGather:
    """Numerical equivalence of the vectorised gather with the loop."""

    @pytest.mark.parametrize("b,c,t,seg", [
        (1, 256, 400, 64),          # production z shape (single batch row)
        (6, 256, 400, 64),
        (12, 256, 320, 320),        # the exact Colab crash shape (B=12)
        (3, 128, 1000, 1),
        (5, 1, 50, 49),
    ])
    def test_matches_reference_loop(self, b, c, t, seg):
        torch.manual_seed(0)
        x = torch.randn(b, c, t)
        ids = torch.randint(0, t - seg + 1, (b,))
        out = commons.slice_segments(x, ids, seg)
        ref = _ref_slice_segments(x, ids, seg)
        assert torch.equal(out, ref)

    @pytest.mark.parametrize("b,t,seg", [
        (12, 320, 320),             # the exact Colab pitchf slice
        (4, 1000, 256),
        (1, 64, 32),
    ])
    def test_slice2_matches_reference_loop(self, b, t, seg):
        torch.manual_seed(1)
        x = torch.randn(b, t)
        ids = torch.randint(0, t - seg + 1, (b,))
        out = commons.slice_segments2(x, ids, seg)
        ref = _ref_slice_segments2(x, ids, seg)
        assert torch.equal(out, ref)

    def test_fp16_precision(self):
        # the training precision on T4-class GPUs — pitchf is fp16 [B, T]
        torch.manual_seed(2)
        x = torch.randn(8, 320).half()
        ids = torch.randint(0, 1, (8,))
        out = commons.slice_segments2(x, ids, 320)
        ref = _ref_slice_segments2(x, ids, 320)
        assert out.dtype == torch.float16
        assert torch.equal(out, ref)

    def test_non_contiguous_input(self):
        # z / mel arriving as transposed views (as in the training loop)
        torch.manual_seed(3)
        base = torch.randn(6, 400, 256)
        x = base.permute(0, 2, 1)               # non-contiguous [b, c, t]
        assert not x.is_contiguous()
        ids = torch.tensor([0, 5, 100, 200, 300, 336])
        out = commons.slice_segments(x, ids, 64)
        ref = _ref_slice_segments(x, ids, 64)
        assert torch.equal(out, ref)

    def test_ids_as_scalar_tensor_rows(self):
        # ids from rand_slice_segments are longs on the input device
        torch.manual_seed(4)
        x = torch.randn(3, 8, 100)
        ids = torch.tensor([0, 50, 92], dtype=torch.long)
        out = commons.slice_segments(x, ids, 8)
        assert out.shape == (3, 8, 8)
        assert torch.equal(out[1], x[1, :, 50:58])


class TestNSFSourceFeed:
    """The GeneratorNSF dim=1 reshape branch is value-identical to transpose."""

    def test_dim1_reshape_equals_transpose(self):
        sine = torch.randn(4, 12800, 1)          # SineGen output (harmonic_num=0)
        t = sine.transpose(1, 2)
        r = sine.reshape(sine.shape[0], 1, sine.shape[1])
        assert torch.equal(t, r)
        assert t.shape == r.shape == (4, 1, 12800)

    def test_dim1_conv_output_equivalent(self):
        # different memory layout -> different float summation order; the
        # criterion is float32 epsilon, not bit identity
        sine = torch.randn(4, 12800, 1)
        conv = torch.nn.Conv1d(1, 32, 1)
        with torch.no_grad():
            ct = conv(sine.transpose(1, 2))
            cr = conv(sine.reshape(4, 1, 12800))
        assert torch.allclose(ct, cr, atol=1e-5, rtol=1e-5)


class TestCompileGuard:
    """Runtime eager-fallback around torch.compile'd networks."""

    class _KernelError(Exception):
        pass

    @staticmethod
    def _fake_kernel_error():
        """An AssertionError whose traceback genuinely runs through a
        torch/_dynamo module frame — what _is_compile_kernel_error detects.
        Crafted by raising inside a helper defined in a real dynamo module:
        importing the frame trick via sys._getframe inside torch's code is
        fragile, so instead we raise with the dynamo module's file patched
        into the traceback via a temporary wrapper."""
        import traceback as _tb

        err = AssertionError("expected size 256==256, stride 1==320 at dim=1")
        # build a synthetic traceback whose frames point at real torch paths
        frames = []
        import torch._dynamo.utils as dyn_utils
        dyn_file = dyn_utils.__file__
        try:
            raise err
        except AssertionError:
            frames = [
                _tb.FrameSummary("trainer_loop.py", 700, "run_step",
                                 lookup_line=False),
                _tb.FrameSummary(dyn_file, 100, "compiled_fn",
                                 lookup_line=False),
            ]
        # StackSummary does not carry __traceback__; emulate via raise from
        # an inductor-flavoured exception: simplest robust approach is to
        # actually raise through a function compiled by torch._dynamo —
        # overkill for a unit test. Instead assert the CLASSIFIER directly
        # on an exception raised inside torch._dynamo's module scope:
        def _raise_inside_dynamo_module():
            # executes with this test file in the traceback BUT the raise
            # site imports nothing; the classifier checks frame FILENAMES —
            # so give it one real dynamo frame by raising from a function
            # we exec with a fake __file__ inside torch/_dynamo:
            import sys
            mod = sys.modules["torch._dynamo.utils"]
            frame_file = mod.__file__
            # exec a raise with globals faked so the traceback frame's
            # filename is the dynamo file
            code = compile("raise AssertionError('kernel boom')",
                           frame_file, "exec")
            exec(code, {})
        try:
            _raise_inside_dynamo_module()
        except AssertionError as exc:
            return exc
        return err

    def test_kernel_error_falls_back_to_eager(self):
        class Raw(torch.nn.Module):
            def forward(self, x):
                return x * 2

        class Compiled:
            def __call__(self, x):
                raise TestCompileGuard._fake_kernel_error()

        raw = Raw()
        guard = Trainer._CompileGuard(Compiled(), raw, "test")
        assert not guard._fell_back
        out = guard(torch.ones(3))
        assert guard._fell_back
        assert out.tolist() == [2, 2, 2]
        # later calls bypass the compiled path entirely
        out2 = guard(torch.ones(3))
        assert out2.tolist() == [2, 2, 2]

    def test_non_kernel_error_propagates(self):
        class Raw(torch.nn.Module):
            def forward(self, x):
                return x

        class Compiled:
            def __call__(self, x):
                raise ValueError("ordinary bug — must reach the caller")

        guard = Trainer._CompileGuard(Compiled(), Raw(), "test")
        with pytest.raises(ValueError):
            guard(torch.ones(3))
        assert not guard._fell_back

    def test_is_compile_kernel_error_classification(self):
        assert Trainer._is_compile_kernel_error(self._fake_kernel_error())
        assert not Trainer._is_compile_kernel_error(ValueError("plain"))
        assert not Trainer._is_compile_kernel_error(
            AssertionError("raised right here"))

    def test_attribute_delegation_and_unwrap(self):
        raw = torch.nn.Linear(4, 4)
        guard = Trainer._CompileGuard(None, raw, "test")
        assert guard.training is True          # nn.Module attr via raw
        guard.eval()
        assert guard.training is False
        assert raw.training is False           # delegated to the SAME module
        assert Trainer._unwrap(guard) is raw
        # plain modules pass through _unwrap unchanged
        assert Trainer._unwrap(raw) is raw


@pytest.mark.slow
class TestRealCompilePath:
    """The REAL torch.compile pass on the f0 generator (CPU is enough: the
    failing assert_size_stride check lives in Inductor's Python wrapper)."""

    def test_compiled_generator_forward(self):
        pytest.importorskip("torch._inductor")
        from rvc_platform.engine.rvc.models import models

        cfg = [1025, 32, 64, 64, 256, 2, 2, 3, 0, "1",
               [3, 7, 11], [[1, 3, 5]] * 3, [10, 10, 2, 2], 64,
               [16, 16, 4, 4], 109, 256]
        net = models.SynthesizerTrnMs768NSFsid(*cfg, is_half=False, sr=40_000)
        net.train()
        torch.manual_seed(0)
        b, phone_t = 2, 40
        spec_t = phone_t * 2
        args = dict(
            phone=torch.randn(b, phone_t, 768),
            phone_lengths=torch.full((b,), phone_t, dtype=torch.long),
            pitch=torch.randint(1, 255, (b, phone_t)),
            pitchf=100 + 50 * torch.rand(b, spec_t),
            y=torch.randn(b, 1025, spec_t),
            y_lengths=torch.full((b,), spec_t, dtype=torch.long),
            ds=torch.zeros(b, dtype=torch.long),
        )
        # NOTE: value comparison against eager is impossible: SineGen draws
        # fresh randn noise on every call, so two runs legitimately differ.
        # The regression is the CRASH itself — running clean through the
        # compiled graph (it raised assert_size_stride here in v1.3.1).
        out_eager = net(**args)
        out = torch.compile(net)(**args)
        assert out[0].shape == out_eager[0].shape
        assert torch.isfinite(out[0]).all(), "compiled output must be finite"


class TestNotebookGpuWheelGuard:
    """Every notebook cell running `uv run` must set UV_NO_SYNC first."""

    def _notebook(self) -> dict:
        nb_path = (Path(__file__).resolve().parents[2] / "notebooks"
                   / "RVC_Training_Colab.ipynb")
        if not nb_path.is_file():
            pytest.skip("notebook not shipped in this layout")
        return json.loads(nb_path.read_text(encoding="utf-8"))

    def test_all_uv_run_cells_guarded(self):
        nb = self._notebook()
        unguarded = []
        for idx, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell["source"])
            if "install_colab.sh" in src:
                continue  # the installer itself manages the env deliberately
            if "uv run" in src and "UV_NO_SYNC" not in src:
                unguarded.append(idx)
        assert not unguarded, f"cells {unguarded} run uv run without the guard"

    def test_training_cell_has_gpu_preflight(self):
        nb = self._notebook()
        training_cells = []
        for cell in nb["cells"]:
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell["source"])
            if "train start" in src and "--experiment" in src:
                training_cells.append(src)
        assert training_cells, "training cell not found"
        for src in training_cells:
            assert "env gpu" in src, \
                "training cell must self-heal CPU-torch before training"
