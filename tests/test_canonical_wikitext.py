from __future__ import annotations

import unittest

import numpy as np

from benchmarks.wikitext103_coverage import (
    EXPECTED_TOKEN_COUNTS,
    OPTIMIZER_STEPS_PER_PASS,
    TRAIN_RECORDS_PER_PASS,
    TRAIN_TARGETS_PER_PASS,
    build_evaluation_index,
    build_pass_permutations,
    build_training_index,
    validate_evaluation_index,
    validate_training_index,
)
from scripts.prepare_canonical_wikitext103 import (
    EXPECTED_INVENTORY_SHA256,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_PAYLOAD_SHA256,
)


class CanonicalWikiTextTests(unittest.TestCase):
    def test_preparation_is_pinned_to_the_recorded_payload(self) -> None:
        self.assertEqual(
            EXPECTED_MANIFEST_SHA256,
            "fc4ef13cbc38070f2d7774dffbfd5be48cab31fe45d6d9995d522fc3bac1dde6",
        )
        self.assertEqual(
            EXPECTED_PAYLOAD_SHA256,
            "f430ce52a43a44b88f5a8ec1ec5882866daaa568595bfbd6b765e3369586f85e",
        )
        self.assertEqual(
            EXPECTED_INVENTORY_SHA256,
            "efa0a0b857d184dccdecf6adafda5d646ec45bae91f01a15cd1165f9fe9ad6b3",
        )

    def test_training_index_covers_every_target_once_per_pass(self) -> None:
        starts, targets = build_training_index(EXPECTED_TOKEN_COUNTS["train"])
        permutations = build_pass_permutations(len(starts), 3, 20_260_818)
        audit = validate_training_index(
            starts,
            targets,
            permutations,
            token_count=EXPECTED_TOKEN_COUNTS["train"],
        )
        self.assertEqual(len(starts), TRAIN_RECORDS_PER_PASS)
        self.assertEqual(len(starts) // 8, OPTIMIZER_STEPS_PER_PASS)
        self.assertEqual(audit["targets_per_pass"], TRAIN_TARGETS_PER_PASS)
        self.assertEqual(
            audit["total_target_presentations"],
            3 * TRAIN_TARGETS_PER_PASS,
        )
        self.assertTrue(
            all(
                np.array_equal(
                    np.sort(permutation),
                    np.arange(len(starts), dtype="<u4"),
                )
                for permutation in permutations
            )
        )

    def test_evaluation_scores_every_transition_once(self) -> None:
        for split in ("validation", "test"):
            index = build_evaluation_index(EXPECTED_TOKEN_COUNTS[split])
            audit = validate_evaluation_index(
                *index,
                token_count=EXPECTED_TOKEN_COUNTS[split],
            )
            self.assertEqual(
                audit["scored_targets"],
                EXPECTED_TOKEN_COUNTS[split] - 1,
            )
            self.assertEqual(audit["maximum_input_tokens"], 2_048)


if __name__ == "__main__":
    unittest.main()
