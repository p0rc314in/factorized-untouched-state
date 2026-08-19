from __future__ import annotations

import unittest

from scripts.verify_results import main


class RecordedResultTests(unittest.TestCase):
    def test_recorded_result_arithmetic_and_provenance(self) -> None:
        main()


if __name__ == "__main__":
    unittest.main()
