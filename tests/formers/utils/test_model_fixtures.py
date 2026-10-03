# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
import tempfile
import unittest
from pathlib import Path

from scripts.unit_test.prepare_model_fixtures import prepare_model_fixtures


class ModelFixtureIsolationTest(unittest.TestCase):
    def test_jobs_share_weights_but_cannot_delete_each_others_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "shared"
            model = source / "PaddleFormers" / "tiny-model"
            model.mkdir(parents=True)
            weight = model / "model.safetensors"
            weight.write_bytes(b"shared weight fixture")
            metadata_name = "flex-ckpt.auto_generated.metadata"
            (model / metadata_name).write_bytes(b"shared cached metadata")
            paths = []
            for job in ["job-a", "job-b"]:
                destination = root / job
                prepare_model_fixtures(source, destination)
                private = destination / "PaddleFormers" / "tiny-model"
                self.assertFalse(private.is_symlink())
                self.assertTrue((private / weight.name).is_symlink())
                self.assertEqual(
                    (private / weight.name).read_bytes(), weight.read_bytes()
                )
                metadata = private / metadata_name
                self.assertFalse(metadata.exists())
                metadata.write_bytes(job.encode())
                paths.append(metadata)
            paths[0].unlink()
            self.assertEqual(paths[1].read_bytes(), b"job-b")
            self.assertEqual(
                (model / metadata_name).read_bytes(), b"shared cached metadata"
            )
            self.assertEqual(weight.read_bytes(), b"shared weight fixture")

    def test_existing_output_and_nested_destinations_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "shared"
            source.mkdir()
            (source / "model.safetensors").write_bytes(b"weights")
            for destination in [source, source / "nested"]:
                with self.assertRaises(ValueError):
                    prepare_model_fixtures(source, destination)
            destination = Path(directory) / "job"
            destination.mkdir()
            (destination / "keep").write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                prepare_model_fixtures(source, destination)
            self.assertEqual((destination / "keep").read_bytes(), b"existing")


if __name__ == "__main__":
    unittest.main()
