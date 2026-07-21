"""Tests for :mod:`euclid_analysis.dataio`."""

from __future__ import annotations

import pytest

from euclid_analysis import dataio


class TestRoundTrip:
    def test_save_then_load_is_identity(self, tmp_path):
        meta = {10: {1: 2, 2: 3}, 20: {1: 1, 3: 4}}
        path = tmp_path / "rt.csv"
        dataio.save_meta_dict(meta, path)
        back = dataio.load_meta_dict(path)
        assert back == meta

    def test_drop_zeros_true_omits_zero_cells(self, tmp_path):
        meta = {10: {1: 2}, 20: {3: 4}}  # tabulation will zero-fill the gaps
        path = tmp_path / "rt.csv"
        dataio.save_meta_dict(meta, path)
        loaded = dataio.load_meta_dict(path, drop_zeros=True)
        assert loaded == meta

    def test_drop_zeros_false_keeps_full_grid(self, tmp_path):
        meta = {10: {1: 2}, 20: {3: 4}}
        path = tmp_path / "rt.csv"
        dataio.save_meta_dict(meta, path)
        loaded = dataio.load_meta_dict(path, drop_zeros=False)
        assert loaded[10] == {1: 2, 3: 0}
        assert loaded[20] == {1: 0, 3: 4}

    def test_outer_keys_are_integers(self, tmp_path):
        meta = {1001: {5: 7}}
        path = tmp_path / "rt.csv"
        dataio.save_meta_dict(meta, path)
        loaded = dataio.load_meta_dict(path)
        assert list(loaded.keys()) == [1001]
        assert all(isinstance(k, int) for k in loaded)


class TestLoadDataset:
    def test_unknown_dataset_raises_keyerror(self):
        with pytest.raises(KeyError):
            dataio.load_dataset("does-not-exist")

    def test_missing_file_raises_filenotfound(self):
        # 'A' (the large gcd-pair table) is not shipped with the repository.
        if (dataio.DATA_DIR / dataio.DATASETS["A"]).exists():
            pytest.skip("dataset 'A' unexpectedly present")
        with pytest.raises(FileNotFoundError):
            dataio.load_dataset("A")

    def test_loads_known_dataset_when_present(self):
        path = dataio.DATA_DIR / dataio.DATASETS["C"]
        if not path.exists():
            pytest.skip("dataset 'C' not present")
        table = dataio.load_dataset("C")
        assert 1001 in table
        assert table[1001][5] == 103591
