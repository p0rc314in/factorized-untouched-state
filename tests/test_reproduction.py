from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import tomllib
import unittest

from scripts.prepare_sdm_runtime import prepare_runtime
from scripts.run_experiments import ARMS, CAMPAIGN, arm_command


ROOT = Path(__file__).resolve().parents[1]


def option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


class ReproductionTests(unittest.TestCase):
    def test_exact_seed_zero_three_arm_matrix(self) -> None:
        self.assertEqual(
            ARMS,
            (
                ("zero", "none", 0),
                ("product", "product_factors", 57_344),
                ("full", "independent_full", 917_504),
            ),
        )

    def test_every_command_uses_the_canonical_balanced_configuration(self) -> None:
        for arm in ARMS:
            command = arm_command(Path("manifest"), Path("output"), arm)
            self.assertEqual(option(command, "--reads"), "16")
            self.assertEqual(option(command, "--writes"), "16")
            self.assertEqual(option(command, "--memory-heads"), "1")
            self.assertEqual(option(command, "--slots"), "1024")
            self.assertEqual(option(command, "--passes"), "3")
            self.assertEqual(option(command, "--schedule-steps"), "21603")
            self.assertEqual(option(command, "--warmup-steps"), "540")
            self.assertEqual(option(command, "--seed"), "0")
            self.assertEqual(option(command, "--arm"), arm[0])
            self.assertEqual(option(command, "--prior-initialization"), arm[1])
            self.assertEqual(option(command, "--initialization-device"), "cuda")
            self.assertEqual(option(command, "--campaign"), CAMPAIGN)

    def test_campaign_matches_the_recorded_experiment(self) -> None:
        self.assertEqual(CAMPAIGN, "factorized-sdm-init-canonical-wt103-seed0-v6")

    def test_public_identity_uses_productkey_init_sdm(self) -> None:
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertEqual(metadata["project"]["name"], "productkey-init-sdm")
        readme = (ROOT / "README.md").read_text()
        self.assertTrue(
            readme.startswith("# Product-Key Initialization for Sparse Delta Memory\n")
        )
        citation = (ROOT / "CITATION.cff").read_text()
        self.assertIn(
            "https://github.com/p0rc314in/productkey-init-sdm",
            citation,
        )
        self.assertNotIn("factorized-untouched-state", citation)

    def test_vendored_released_sdm_tree_is_unmodified(self) -> None:
        source_root = ROOT / "third_party/released_sdm"
        manifest = json.loads((source_root / "SOURCE.json").read_text())
        observed = {
            relative: hashlib.sha256((source_root / relative).read_bytes()).hexdigest()
            for relative in sorted(manifest["files"])
        }
        self.assertFalse(manifest["modified"])
        self.assertEqual(observed, manifest["files"])

    def test_recorded_loader_patch_applies_without_changing_kernel_sources(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test-sdm-runtime-") as temporary:
            runtime = prepare_runtime(Path(temporary) / "sdm_patched")
            manifest = json.loads((runtime / "RUNTIME_SOURCE.json").read_text())
            self.assertFalse(manifest["model_and_kernel_mechanics_changed"])
            self.assertEqual(
                manifest["loader_patch_sha256"],
                "15751efa855485d3c2eabbaa8b0c686f94d11ee03d64366dc8d2cd6e7d36f574",
            )

    def test_note_frames_the_recorded_sdm_tradeoff(self) -> None:
        readme = (ROOT / "README.md").read_text()
        self.assertIn("## Intuition", readme)
        self.assertIn("Product-key factorization gives a third option", readme)
        self.assertIn("Θ(Ndᵥ)", readme)
        self.assertIn("Θ(√N dᵥ)", readme)
        self.assertIn("eight-layer model", readme)
        self.assertIn("about 15M trainable parameters", readme)
        self.assertIn("353,941,347 target presentations per arm", readme)
        self.assertIn("factorized initial memory roughly matched", readme)
        self.assertIn("Zero initialization removes that term", readme)
        self.assertNotIn("parameter-scaling blocker", readme)


if __name__ == "__main__":
    unittest.main()
