"""Small constructed arrays test mechanics, never research results."""

import numpy as np
import pandas as pd
import pytest
import torch

from tracast.data import Scaler, Windows, missing_report, split, validate
from tracast.experiment import Config, evaluate, run, seed_all
from tracast.model import GRU, Metrics


@pytest.fixture
def frame():
    return pd.DataFrame(
        {"a": np.arange(300, dtype=float) + 1, "b": np.full(300, 50.0)},
        index=pd.date_range("2012-03-01", periods=300, freq="5min"),
    )


@pytest.mark.parametrize(
    "kind",
    ["duplicate", "reverse", "gap", "negative", "infinity", "text", "sensor", "nat"],
)
def test_schema_rejects_bad_data(frame, kind):
    if kind == "duplicate":
        frame.index = [frame.index[0]] * len(frame)
        frame.index = pd.DatetimeIndex(frame.index)
    elif kind == "reverse":
        frame = frame.iloc[::-1]
    elif kind == "gap":
        frame = frame.drop(frame.index[3])
    elif kind == "negative":
        frame.iloc[0, 0] = -1
    elif kind == "infinity":
        frame.iloc[0, 0] = np.inf
    elif kind == "text":
        frame["a"] = "bad"
    elif kind == "sensor":
        frame.columns = ["a", "a"]
    else:
        frame.index = pd.DatetimeIndex([pd.NaT] + list(frame.index[1:]))
    with pytest.raises(ValueError):
        validate(frame)


def test_missing_counts(frame):
    frame.iloc[0, 0] = 0
    frame.iloc[1, 0] = np.nan
    validate(frame)
    report = missing_report(frame)
    assert report["zero_count"] == 1
    assert report["nan_count"] == 1
    assert report["missing_fraction"] == pytest.approx(2 / 600)


def test_splits_and_target_boundaries(frame):
    parts = split(frame, 12, 12)
    assert [len(part) for part in parts.values()] == [210, 30, 60]
    assert parts["train"].index[-1] < parts["val"].index[0] < parts["test"].index[0]
    scaler = Scaler().fit(parts["train"])
    for part in parts.values():
        ds = Windows(part, scaler)
        assert len(ds) == len(part) - 23
        first_x, first_y, mask = ds[0]
        assert first_x.shape == (12, 2) and first_y.shape == (12, 2)
        np.testing.assert_array_equal(first_y, part.iloc[12:24])
        np.testing.assert_array_equal(ds[len(ds) - 1][1], part.iloc[-12:])
        assert mask.all()


def test_scaler_train_only_and_causal_fill(frame):
    parts = split(frame)
    train = parts["train"].copy()
    train.iloc[0, 0] = 0
    scaler = Scaler().fit(train)
    assert scaler.mean[0] == pytest.approx(train.a.iloc[1:].mean())
    assert scaler.std[1] == 1
    mean_before = scaler.mean.copy()
    held = parts["val"].copy()
    held.iloc[:2, 0] = 0
    held.iloc[3, 0] = np.nan
    x = scaler.inputs(held)
    assert x[0, 0] == 0 and x[1, 0] == 0
    assert x[3, 0] == x[2, 0]
    held.iloc[10:, :] = 10000
    np.testing.assert_array_equal(scaler.inputs(held)[:10], x[:10])
    np.testing.assert_array_equal(scaler.mean, mean_before)


def test_all_missing_sensor_fails(frame):
    frame["a"] = 0
    with pytest.raises(ValueError, match="no observed"):
        Scaler().fit(frame)


def test_missing_targets_stay_masked(frame):
    frame.iloc[12, 0] = 0
    frame.iloc[13, 1] = np.nan
    ds = Windows(frame, Scaler().fit(frame))
    x, y, mask = ds[0]
    assert torch.isfinite(x).all() and torch.isfinite(y).all()
    assert not mask[0, 0] and not mask[1, 1]


def test_metrics_count_weighting():
    scores = Metrics(2)
    target = torch.zeros(2, 2, 1)
    pred = torch.tensor([[[1.0], [3.0]], [[2.0], [100.0]]])
    mask = torch.tensor([[[True], [True]], [[True], [False]]])
    scores.update(pred[:1], target[:1], mask[:1])
    scores.update(pred[1:], target[1:], mask[1:])
    result = scores.result()
    assert result["mae"] == 2.0
    assert result["rmse"] == pytest.approx(np.sqrt(14 / 3))
    assert result["per_horizon"][1]["count"] == 1


def test_empty_metric_fails():
    with pytest.raises(ValueError, match="No observed"):
        Metrics(2).result()


def test_persistence_is_last_observation(frame):
    scaler = Scaler().fit(frame)
    ds = Windows(frame, scaler)
    loader = torch.utils.data.DataLoader(ds, batch_size=17)
    result = evaluate(
        None, loader, torch.from_numpy(scaler.mean), torch.from_numpy(scaler.std), 12
    )
    # Sensor a rises by one each step; sensor b is constant.
    assert result["mae"] == pytest.approx(3.25, abs=1e-5)
    assert result["per_horizon"][-1]["mae"] == pytest.approx(6.0, abs=1e-5)


def test_gru_backward_and_seed():
    seed_all(42, 1)
    first = GRU(2, 8, 3)
    x = torch.ones(4, 12, 2)
    result = first(x)
    assert result.shape == (4, 3, 2)
    result.square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in first.parameters()
    )
    seed_all(42, 1)
    second = GRU(2, 8, 3)
    assert torch.equal(result, second(x))


@pytest.mark.parametrize(
    "changes", [{"epochs": 0}, {"history": -1}, {"lr": float("nan")}, {"rows": 0}]
)
def test_config_validation(changes):
    with pytest.raises(ValueError):
        Config(**changes).validate()


def test_too_short_split(frame):
    with pytest.raises(ValueError, match="Each chronological split"):
        split(frame.iloc[:100])


def test_end_to_end_repeats(frame, tmp_path, monkeypatch):
    import tracast.experiment as experiment

    monkeypatch.setattr(experiment, "load", lambda *args: frame.astype("float32"))
    config = Config(sensors=2, history=3, horizon=2, hidden=4, epochs=1, threads=1)
    first = run(config, "test-fixture-only", tmp_path / "first")
    second = run(config, "test-fixture-only", tmp_path / "second")
    assert first == second
    assert (tmp_path / "first/status.json").read_text() == (
        tmp_path / "second/status.json"
    ).read_text()
    with pytest.raises(FileExistsError):
        run(config, "test-fixture-only", tmp_path / "first")


def test_fetch_rejects_corruption(tmp_path, monkeypatch):
    import io
    import tracast.data as data

    path = tmp_path / "metr-la.h5"
    monkeypatch.setattr(
        data.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"not hdf5")
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        data.fetch(path)
    assert not path.exists() and not path.with_suffix(".partial").exists()
    path.write_bytes(b"wrong existing file")
    with pytest.raises(ValueError, match="Existing dataset checksum"):
        data.fetch(path)


def test_load_requires_verified_file(tmp_path):
    from tracast.data import load

    path = tmp_path / "bad.h5"
    path.write_bytes(b"not the dataset")
    with pytest.raises(ValueError, match="checksum"):
        load(path)


def test_legacy_hdf_frequency_metadata(monkeypatch):
    import tracast.data as data

    frame = pd.DataFrame(
        np.ones((34272, 207), dtype="float32"),
        index=pd.date_range("2012-03-01", periods=34272, freq="5min"),
    )
    # Emulate the metadata stored by the original old-pandas HDF5 writer.
    frame.index._data._freq = b"legacy serialized pandas offset"
    monkeypatch.setattr(data, "digest", lambda path: data.SHA256)
    monkeypatch.setattr(data.pd, "read_hdf", lambda *args, **kwargs: frame)
    selected = data.load("test-fixture-only", sensors=8, rows=2016)
    assert selected.shape == (2016, 8)
    assert selected.index.freq is None
    assert selected.index.inferred_freq == "5min"
