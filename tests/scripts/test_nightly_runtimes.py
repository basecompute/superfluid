"""Tests for scripts/nightly_runtimes.py: how a llama.cpp build that passed
the nightly is recorded, on made-up releases.

    python3 -m unittest discover -s tests/scripts -v

Nothing is downloaded.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import nightly_runtimes  # noqa: E402


def published(version: str, *names: str) -> list[dict]:
    """A release's assets as GitHub lists them, each with a digest of its own."""
    return [{"name": n.format(v=version), "size": i + 1, "digest": f"sha256:{i:064x}"} for i, n in enumerate(names)]


def tested(version: str, *names: str) -> dict:
    return {"version": version, "assets": [{"name": n.format(v=version), "size": 1, "sha256": "0" * 64} for n in names]}


PREVIOUS = tested(
    "b100",
    "cudart-llama-{v}-bin-ubuntu-cuda-12.8-x64.tar.gz",
    "cudart-llama-{v}-bin-ubuntu-cuda-13.4-x64.tar.gz",
    "llama-{v}-bin-macos-arm64.tar.gz",
    "llama-{v}-bin-ubuntu-cuda-12.8-x64.tar.gz",
    "llama-{v}-bin-ubuntu-cuda-13.4-x64.tar.gz",
    "llama-{v}-bin-ubuntu-rocm-10.0-x64.tar.gz",
    "llama-{v}-bin-ubuntu-x64.tar.gz",
)


class CarriedAssets(unittest.TestCase):
    def names(self, assets: list[dict]) -> list[str]:
        return [a["name"] for a in assets]

    def test_the_same_names_under_the_new_build(self):
        release = published("b101", *(a["name"].replace("b100", "{v}") for a in PREVIOUS["assets"]))
        assets, missing, moved = nightly_runtimes.carried_assets(PREVIOUS, "b101", release)
        self.assertEqual(self.names(assets), [a["name"].replace("b100", "b101") for a in PREVIOUS["assets"]])
        self.assertEqual((missing, moved), ([], []))
        self.assertTrue(all(len(a["sha256"]) == 64 and a["size"] > 0 for a in assets))

    def test_a_backend_that_moved_its_minor_version_is_followed(self):
        # b12184: llama.cpp's CUDA 13 build went from cuda-13.4 to cuda-13.5, its runtime with it.
        release = published(
            "b101",
            "cudart-llama-{v}-bin-ubuntu-cuda-12.8-x64.tar.gz",
            "cudart-llama-{v}-bin-ubuntu-cuda-13.5-x64.tar.gz",
            "llama-{v}-bin-macos-arm64.tar.gz",
            "llama-{v}-bin-ubuntu-cuda-12.8-x64.tar.gz",
            "llama-{v}-bin-ubuntu-cuda-13.5-x64.tar.gz",
            "llama-{v}-bin-ubuntu-rocm-10.0-x64.tar.gz",
            "llama-{v}-bin-ubuntu-x64.tar.gz",
            "llama-{v}-bin-ubuntu-openvino-2026.4.1-x64.tar.gz",
        )
        assets, missing, moved = nightly_runtimes.carried_assets(PREVIOUS, "b101", release)
        names = self.names(assets)
        self.assertIn("llama-b101-bin-ubuntu-cuda-13.5-x64.tar.gz", names)
        self.assertIn("cudart-llama-b101-bin-ubuntu-cuda-13.5-x64.tar.gz", names)
        self.assertIn("llama-b101-bin-ubuntu-cuda-12.8-x64.tar.gz", names, "CUDA 12 is a slot of its own")
        self.assertNotIn("llama-b101-bin-ubuntu-openvino-2026.4.1-x64.tar.gz", names, "only what the previous build carried")
        self.assertEqual(len(names), len(PREVIOUS["assets"]))
        self.assertEqual(missing, [])
        self.assertEqual(
            moved,
            [
                "cudart-llama-b101-bin-ubuntu-cuda-13.4-x64.tar.gz is now cudart-llama-b101-bin-ubuntu-cuda-13.5-x64.tar.gz",
                "llama-b101-bin-ubuntu-cuda-13.4-x64.tar.gz is now llama-b101-bin-ubuntu-cuda-13.5-x64.tar.gz",
            ],
        )

    def test_a_major_version_is_not_another_slot_s_minor(self):
        # CUDA 13 gone and CUDA 14 come: that is a new backend for a human to add, not a move.
        release = published(
            "b101",
            "llama-{v}-bin-macos-arm64.tar.gz",
            "llama-{v}-bin-ubuntu-cuda-12.8-x64.tar.gz",
            "llama-{v}-bin-ubuntu-cuda-14.0-x64.tar.gz",
            "llama-{v}-bin-ubuntu-x64.tar.gz",
        )
        assets, missing, moved = nightly_runtimes.carried_assets(PREVIOUS, "b101", release)
        self.assertNotIn("llama-b101-bin-ubuntu-cuda-14.0-x64.tar.gz", self.names(assets))
        self.assertIn("llama-b101-bin-ubuntu-cuda-13.4-x64.tar.gz", missing)
        self.assertEqual(moved, [])

    def test_an_archive_is_listed_once_when_two_old_ones_land_on_it(self):
        previous = tested("b100", "llama-{v}-bin-ubuntu-cuda-13.4-x64.tar.gz", "llama-{v}-bin-ubuntu-cuda-13.5-x64.tar.gz")
        release = published("b101", "llama-{v}-bin-ubuntu-cuda-13.5-x64.tar.gz")
        assets, missing, moved = nightly_runtimes.carried_assets(previous, "b101", release)
        self.assertEqual(self.names(assets), ["llama-b101-bin-ubuntu-cuda-13.5-x64.tar.gz"])
        self.assertEqual((missing, len(moved)), ([], 1))

    def test_of_several_minors_the_newest_takes_the_slot(self):
        release = published("b101", "llama-{v}-bin-ubuntu-cuda-13.10-x64.tar.gz", "llama-{v}-bin-ubuntu-cuda-13.9-x64.tar.gz")
        assets, _, _ = nightly_runtimes.carried_assets(tested("b100", "llama-{v}-bin-ubuntu-cuda-13.4-x64.tar.gz"), "b101", release)
        self.assertEqual(self.names(assets), ["llama-b101-bin-ubuntu-cuda-13.10-x64.tar.gz"], "compared as numbers")

    def test_an_asset_with_no_sha256_is_missing(self):
        release = published("b101", "llama-{v}-bin-macos-arm64.tar.gz")
        release[0]["digest"] = None
        assets, missing, _ = nightly_runtimes.carried_assets(tested("b100", "llama-{v}-bin-macos-arm64.tar.gz"), "b101", release)
        self.assertEqual((assets, missing), ([], ["llama-b101-bin-macos-arm64.tar.gz"]))


if __name__ == "__main__":
    unittest.main()
