import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_continual_baseline as runner


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        enable_thinking=None,
        add_generation_prompt=False,
    ):
        assert tokenize is True
        assert enable_thinking is False
        prompt = [101, 102, 103]
        if messages[-1]["role"] == "assistant":
            answer = [20 + (ord(char) % 70) for char in messages[-1]["content"]]
            return {"input_ids": prompt + answer + [self.eos_token_id], "attention_mask": [1] * (len(prompt) + len(answer) + 1)}
        return {"input_ids": prompt, "attention_mask": [1] * len(prompt)}


def record(answer):
    return {
        "task_id": "dummy_task",
        "instance_id": "dummy-1",
        "definition": "Do the requested task.",
        "input": "Example input.",
        "targets": [answer],
    }


class ContinualBaselineTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = DummyTokenizer()

    def test_assistant_only_labels_include_eos(self):
        encoded = runner.encode_training_record(self.tokenizer, record("ok"), max_length=32)
        self.assertEqual(encoded["labels"][:3], [-100, -100, -100])
        self.assertEqual(encoded["labels"][3:], encoded["input_ids"][3:])
        self.assertEqual(encoded["input_ids"][-1], self.tokenizer.eos_token_id)

    def test_long_answer_is_truncated_and_keeps_eos(self):
        encoded = runner.encode_training_record(self.tokenizer, record("abcdef"), max_length=3)
        self.assertEqual(len(encoded["input_ids"]), 3)
        self.assertEqual(encoded["input_ids"][-1], self.tokenizer.eos_token_id)
        self.assertEqual(encoded["labels"], encoded["input_ids"])

    def test_batch_padding_masks_attention_and_loss(self):
        batch = runner.make_batch(
            self.tokenizer,
            [record("x"), record("long")],
            max_length=32,
            device=torch.device("cpu"),
        )
        self.assertEqual(batch["input_ids"].shape[0], 2)
        first_length = int(batch["attention_mask"][0].sum().item())
        self.assertTrue(torch.all(batch["attention_mask"][0, :first_length] == 1))
        self.assertTrue(torch.all(batch["attention_mask"][0, first_length:] == 0))
        self.assertTrue(torch.all(batch["labels"][0, first_length:] == -100))

    def test_normalized_exact_match_and_oom_detection(self):
        self.assertEqual(runner.normalise_answer(" A.\n"), "a")
        self.assertTrue(runner.is_cuda_oom(RuntimeError("CUDA out of memory")))
        self.assertFalse(runner.is_cuda_oom(RuntimeError("unrelated failure")))


if __name__ == "__main__":
    unittest.main()
