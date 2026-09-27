"""Semantic and lifecycle checks on CUDA; no pre-captured artifact fixtures."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest

import torch
from torch.nn import functional as F

from study02.examples import ManualLayerNorm, mutate_view
from study02.model import GPT, GPTConfig
from study02.notebook_utils import run_experiment
from study02.runtime import REPO_DIR, cuda_environment


class CUDASemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cuda_environment()  # Fail explicitly if the required execution target is absent.
        torch.backends.cuda.matmul.allow_tf32 = False

    def setUp(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    def test_model_shapes_weight_tying_and_causality(self):
        model = GPT().cuda().eval()
        tokens = torch.randint(128, (1, 8), device="cuda")
        targets = tokens.clone()
        targets[:, -1] = -1
        with torch.no_grad():
            inference, no_loss = model(tokens)
            logits, loss = model(tokens, targets)
            perturbed = tokens.clone()
            perturbed[:, -1] = (perturbed[:, -1] + 1) % 128
            changed, _ = model(perturbed, targets)
        self.assertEqual(tuple(inference.shape), (1, 1, 128))
        self.assertEqual(tuple(logits.shape), (1, 8, 128))
        self.assertIsNone(no_loss)
        self.assertTrue(torch.isfinite(loss))
        self.assertIs(model.transformer.wte.weight, model.lm_head.weight)
        torch.testing.assert_close(inference[:, 0], logits[:, -1])
        torch.testing.assert_close(changed[:, :-1], logits[:, :-1])

    def test_manual_sdpa_same_weights_forward_and_gradients(self):
        manual = GPT(GPTConfig(attention_impl="manual")).cuda()
        sdpa = GPT(GPTConfig(attention_impl="sdpa")).cuda()
        state = {k: v for k, v in manual.state_dict().items() if k != "transformer.h.0.attn.bias"}
        sdpa.load_state_dict(state, strict=True)
        x = torch.randint(128, (1, 8), device="cuda")
        targets = torch.randint(128, (1, 8), device="cuda")
        y1, loss1 = manual(x, targets)
        y2, loss2 = sdpa(x, targets)
        loss1.backward(); loss2.backward()
        torch.testing.assert_close(y1, y2, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(loss1, loss2)
        for (name1, p1), (name2, p2) in zip(manual.named_parameters(), sdpa.named_parameters()):
            self.assertEqual(name1, name2)
            torch.testing.assert_close(p1.grad, p2.grad, rtol=1e-4, atol=1e-5)

    def test_layernorm_uses_same_variance_and_affine(self):
        layer = ManualLayerNorm(7).cuda()
        with torch.no_grad():
            layer.weight.normal_(); layer.bias.normal_()
        x = torch.randn(2, 3, 7, device="cuda", requires_grad=True)
        actual = layer(x)
        expected = F.layer_norm(x, (7,), layer.weight, layer.bias, 1e-5)
        torch.testing.assert_close(actual, expected)
        g1 = torch.autograd.grad(actual.sum(), x, retain_graph=True)[0]
        g2 = torch.autograd.grad(expected.sum(), x)[0]
        torch.testing.assert_close(g1, g2, atol=1e-5, rtol=1e-4)

    def test_functionalization_preserves_input_mutation(self):
        a = torch.zeros(2, 3, device="cuda")
        b = a.clone()
        out_a = mutate_view(a)
        out_b = torch.func.functionalize(mutate_view)(b)
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(out_a, out_b)
        torch.testing.assert_close(b, torch.ones_like(b))


class FreshRuns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cuda_environment()

    def test_same_experiment_creates_distinct_processes_and_results(self):
        first = run_experiment("python_runtime")
        second = run_experiment("python_runtime")
        self.assertNotEqual(first["run_id"], second["run_id"])
        manifests = [json.loads((Path(r["_run_dir"]) / "manifest.json").read_text()) for r in (first, second)]
        self.assertNotEqual(manifests[0]["pid"], manifests[1]["pid"])
        for manifest in manifests:
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(Path(manifest["environment"]["executable"]).resolve(), Path(sys.executable).resolve())
        self.assertEqual(first["tiny_output"], second["tiny_output"])
        result_path = Path(first["_run_dir"]) / "result.json"
        original = result_path.read_bytes()
        attempted = subprocess.run([sys.executable, "-m", "study02.experiments.python_runtime",
                                    "--output", first["_run_dir"]], cwd=REPO_DIR, capture_output=True, text=True)
        self.assertNotEqual(attempted.returncode, 0)
        self.assertEqual(original, result_path.read_bytes())

    def test_child_failure_is_not_replaced_by_a_previous_result(self):
        with self.assertRaisesRegex(RuntimeError, "All shapes must be positive"):
            run_experiment("inductor", workload="gelu", channels=-1)

    def test_current_fused_ir_excludes_nested_scheduler_nodes(self):
        result = run_experiment("inductor", workload="residual_layernorm", channels=16)
        capture = result["captures"][0]
        pre_ids = {n["id"] for n in capture["pre"]["nodes"]}
        post_ids = {n["id"] for n in capture["post"]["nodes"]}
        groups = [n for n in capture["post"]["nodes"] if len(n["members"]) > 1]
        self.assertTrue(groups, "This pinned small CUDA reduction should expose a fused group")
        for group in groups:
            self.assertTrue(set(group["members"]) <= pre_ids)
            self.assertFalse(set(group["members"]) & post_ids)
            self.assertTrue(group["reads"], "Pretty-printed multiline dependencies must be parsed")
        self.assertTrue(capture["wrapper"]["buffers"], "Allocations under DeviceGuard must be read")
        for buffer in capture["wrapper"]["buffers"]:
            self.assertLessEqual(buffer["first_line"], buffer["last_line"])
        self.assertTrue(result["correctness"]["passed"])


if __name__ == "__main__":
    unittest.main()
