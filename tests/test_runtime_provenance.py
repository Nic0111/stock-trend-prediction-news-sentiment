from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pipeline.runtime_provenance import sha256_file


class RuntimeProvenanceTests(unittest.TestCase):
    def test_cached_hash_invalidates_when_file_state_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            path.write_bytes(b"first")
            first = sha256_file(path)
            self.assertEqual(sha256_file(path), first)

            path.write_bytes(b"second-version")
            second = sha256_file(path)

        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
