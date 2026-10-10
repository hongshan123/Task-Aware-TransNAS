from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from scripts.run_phase6a_localization import (  # noqa: E402
    ActivationHooks,
    _param_candidates,
    _param_keys,
    _score_logits,
    activation_candidates,
)


class _FakeTokenizer:
    unk_token_id = 0

    def encode(self, text, add_special_tokens=False):
        mapping = {"Yes": [1], " Yes": [2], "No": [3], " No": [4]}
        return mapping.get(text, [5])


class _FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.o_proj = nn.Identity()


class _FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _FakeAttention()
        self.mlp = nn.Identity()

    def forward(self, x):
        return x


class _FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Identity()
        self.layers = nn.ModuleList([_FakeLayer()])
        self.norm = nn.Identity()


class _FakeCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _FakeBackbone()
        self.lm_head = nn.Identity()


class Phase6ALocalizationTests(unittest.TestCase):
    def test_activation_sweep_covers_all_28_layers_and_configured_groups(self):
        candidates = activation_candidates(28)
        self.assertEqual(len(candidates), 361)
        for component in ("attention_o_proj_output", "mlp_output", "transformer_block_output"):
            singles = [row for row in candidates if row["component"] == component and row["scope"] == "single_layer"]
            groups = [row for row in candidates if row["component"] == component and row["scope"] == "contiguous_group"]
            self.assertEqual(len(singles), 28)
            self.assertEqual({row["layers"][0] for row in singles}, set(range(28)))
            self.assertEqual(len(groups), 89)
            self.assertEqual({len(row["layers"]) for row in groups}, {2, 4, 8, 14, 28})
            for row in groups:
                self.assertEqual(row["layers"], list(range(row["layers"][0], row["layers"][-1] + 1)))

    def test_parameter_sweep_has_single_group_and_global_scopes(self):
        candidates = _param_candidates(28)
        self.assertEqual(len(candidates), 560)
        self.assertEqual(sum(row["scope"] == "single_layer" for row in candidates), 84)
        self.assertEqual(sum(row["scope"] == "contiguous_group" for row in candidates), 267)
        self.assertEqual(sum(row["scope"] == "single_projection_layer" for row in candidates), 196)
        self.assertEqual(sum(row["scope"] == "global_projection" for row in candidates), 7)
        globals_ = {row["candidate_id"] for row in candidates if row["scope"] == "global"}
        self.assertEqual(globals_, {
            "attention_global", "mlp_global", "attention_plus_mlp_global",
            "norm_global", "embedding_lm_head_global", "backbone_global",
        })

    def test_parameter_keys_respect_component_scope_and_tied_embedding(self):
        names = [
            f"model.layers.{layer}.self_attn.{projection}.weight"
            for layer in range(3) for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
        ]
        names += [
            f"model.layers.{layer}.mlp.{projection}.weight"
            for layer in range(3) for projection in ("gate_proj", "up_proj", "down_proj")
        ]
        names += [f"model.layers.{layer}.{norm}.weight" for layer in range(3) for norm in ("input_layernorm", "post_attention_layernorm")]
        names += ["model.norm.weight", "model.embed_tokens.weight"]
        parameters = {name: nn.Parameter(torch.zeros(1)) for name in names}

        attention = _param_keys(parameters, {"component": "attention", "layers": [1]})
        self.assertEqual(len(attention), 4)
        self.assertTrue(all("model.layers.1.self_attn." in name for name in attention))

        mlp = _param_keys(parameters, {"component": "mlp", "layers": [0, 2]})
        self.assertEqual(len(mlp), 6)
        self.assertTrue(all(name.startswith(("model.layers.0.mlp.", "model.layers.2.mlp.")) for name in mlp))

        norms = _param_keys(parameters, {"component": "norm", "layers": None})
        self.assertEqual(len(norms), 7)
        tied = _param_keys(parameters, {"component": "embedding_lm_head", "layers": None})
        self.assertEqual(tied, ["model.embed_tokens.weight"])
        q_projection = _param_keys(parameters, {"component": "attention_q_proj", "layers": [2]})
        gate_projection = _param_keys(parameters, {"component": "mlp_gate_proj", "layers": [1]})
        self.assertEqual(q_projection, ["model.layers.2.self_attn.q_proj.weight"])
        self.assertEqual(gate_projection, ["model.layers.1.mlp.gate_proj.weight"])

    def test_activation_hook_patches_only_the_requested_suffix(self):
        model = _FakeCausalLM()
        hooks = ActivationHooks(model, "patch")
        hooks.start = 1
        hooks.bank = {("attention_o_proj_output", 0): torch.tensor([[4.0, 5.0], [6.0, 7.0]])}
        hooks.candidates = [{
            "candidate_id": "attention_layer_0", "component": "attention_o_proj_output",
            "layers": [0], "scope": "single_layer", "operation": "swap",
        }]
        try:
            source = torch.zeros((1, 3, 2))
            output = model.model.layers[0].self_attn.o_proj(source)
            torch.testing.assert_close(output[:, 0], torch.zeros((1, 2)))
            torch.testing.assert_close(output[0, 1:], hooks.bank[("attention_o_proj_output", 0)])
        finally:
            hooks.close()

    def test_score_logits_reports_finite_nll_and_yes_no_margin(self):
        logits = torch.zeros((1, 2, 8))
        logits[0, 0, 1] = 2.0
        logits[0, 0, 2] = 1.0
        logits[0, 0, 3] = -1.0
        logits[0, 0, 4] = -2.0
        packs = [{
            "target_ids": [1, 5],
            "task_id": "task1501_dstc3_answer_generation",
            "record": {"instance_id": "fixture"},
        }]
        result = _score_logits(logits, packs, _FakeTokenizer())[0]
        self.assertEqual(result["token_count"], 2)
        self.assertGreater(result["yes_minus_no_log_probability_margin"], 0)
        self.assertTrue(torch.isfinite(torch.tensor(result["nll_sum"])))


if __name__ == "__main__":
    unittest.main()
