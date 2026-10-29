import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from . import LABELS, LOADERS, SignalReader, get_label_by_idx, label_to_vector
from .loaders import _read_apnea_annotation


class RegistryContractTests(unittest.TestCase):
    def test_builtin_registries_and_round_trip(self):
        self.assertEqual(set(LABELS), set(LOADERS))
        for dataset, labels in LABELS.items():
            self.assertEqual({value["label"] for value in labels.values()}, set(range(len(labels))))
            for acronym, spec in labels.items():
                vector = label_to_vector(dataset, acronym)
                self.assertEqual(sum(vector), 1)
                self.assertEqual(get_label_by_idx(dataset, spec["label"]), acronym)

    def test_ptbxl_loader_is_descriptor_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ptbxl_database.csv").write_text(
                "ecg_id,filename_hr,filename_lr,scp_codes,age,sex,strat_fold\n"
                "1,records100/00001,records100/00001,\"{'NORM': 100}\",50,M,1\n",
                encoding="utf-8",
            )
            record = root / "records100/00001"
            record.parent.mkdir()
            record.with_suffix(".hea").write_text("12 100 1\n", encoding="ascii")
            record.with_suffix(".dat").write_bytes(b"")
            result = next(LOADERS["ptbxl"](root))
            self.assertEqual(result["file_path"], str(root / "records100/00001"))
            self.assertEqual(set(result), {"file_path", "label", "meta"})
            self.assertEqual(result["label"], "NORM")
            self.assertEqual(result["meta"]["label_raw"], {"NORM": 100})

    def test_header_loaders_require_known_labels_and_signal_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            normal = root / "normal"
            normal.with_suffix(".hea").write_text("12 500 1\n#Dx: NORM\n", encoding="ascii")
            normal.with_suffix(".dat").write_bytes(b"")
            unknown = root / "unknown"
            unknown.with_suffix(".hea").write_text("12 500 1\n#Dx: UNKNOWN\n", encoding="ascii")
            missing = root / "missing"
            missing.with_suffix(".hea").write_text("12 500 1\n#Dx: NORMAL\n", encoding="ascii")

            ptbdb = list(LOADERS["ptbdb"](root))
            ludb = list(LOADERS["ludb"](root))
            self.assertEqual([item["label"] for item in ptbdb], ["NORMAL"])
            self.assertEqual([item["label"] for item in ludb], ["NORMAL"])
            self.assertEqual(ptbdb[0]["file_path"], str(normal))

    def test_challenge2020_maps_snomed_codes_and_requires_mat_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapped = root / "mapped"
            mapped.with_suffix(".hea").write_text("mapped 12 500\n#Dx: 426783006, 164889003\n", encoding="ascii")
            mapped.with_suffix(".mat").write_bytes(b"")
            missing = root / "missing"
            missing.with_suffix(".hea").write_text("missing 12 500\n#Dx: 426783006\n", encoding="ascii")
            unknown = root / "unknown"
            unknown.with_suffix(".hea").write_text("unknown 12 500\n#Dx: 999999999\n", encoding="ascii")

            records = list(LOADERS["challenge2020"](root))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["label"], "AF")
            self.assertEqual(records[0]["meta"]["n_sig"], 12)

    def test_challenge2018_uses_apn_sidecar_and_honors_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, annotation in (("record_a", "N A A"), ("record_n", "N N")):
                record = root / name
                record.with_suffix(".hea").write_text("1 100 1\n", encoding="ascii")
                record.with_suffix(".mat").write_bytes(b"")
                record.with_suffix(".apn").write_text(annotation, encoding="ascii")

            records = list(LOADERS["challenge2018"](root, limit=1))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["label"], "APNEA")

    def test_challenge2018_reads_binary_annotations_through_wfdb(self):
        with tempfile.TemporaryDirectory() as directory:
            annotation = Path(directory) / "record.apn"
            annotation.write_bytes(b"not text A N")
            fake_wfdb = SimpleNamespace(rdann=lambda *_: SimpleNamespace(symbol=["N", "A", "A"]))
            with patch.dict(sys.modules, {"wfdb": fake_wfdb}):
                self.assertEqual(_read_apnea_annotation(annotation), ["N", "A", "A"])

    def test_signal_reader_rejects_incomplete_wfdb_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            record = Path(directory) / "incomplete"
            record.with_suffix(".hea").write_text("12 500 1\n", encoding="ascii")
            with self.assertRaises(FileNotFoundError):
                SignalReader.metadata(record)


if __name__ == "__main__":
    unittest.main()
