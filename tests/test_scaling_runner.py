"""The scaling runner on fake benchmark folders: no training involved."""

import json

import pytest

from tracast import scaling


def fake_trial(model, seed, mae, core=None):
    horizons = [
        {"minutes": 5 * (i + 1), "mae": mae + 0.1 * i, "rmse": mae + 1}
        for i in range(3)
    ]
    test = {"mae": mae, "rmse": mae + 1, "per_horizon": horizons}
    if core is not None:
        test["core"] = {"16": {"mae": core, "rmse": core + 1, "per_horizon": horizons}}
    return {
        "model": model,
        "seed": seed,
        "test": test,
        "history": [{}] * (0 if model in ("ha", "persistence") else 4),
        "parameter_count": 0 if model in ("ha", "persistence") else 4000,
        "train_seconds": 10.0 + seed,
        "inference": {"median_ms_per_window": 0.5},
        "peak_memory": {"bytes": 300 * 1024**2},
    }


def write_size(folder, count, status="complete"):
    folder.mkdir(parents=True)
    results = []
    for model, base in (("ha", 3.7), ("sgru_time", 3.2), ("gru_time", 3.1)):
        for seed in (7,) if model == "ha" else (7, 42):
            core = None if count == 16 else base + 0.01 * count / 16
            results.append(fake_trial(model, seed, base + 0.02 * count / 16, core))
    (folder / "results.json").write_text(json.dumps(results))
    (folder / "status.json").write_text(json.dumps({"status": status}))


def test_load_config_checks_the_sizes(tmp_path):
    good = {"sensor_counts": [16, 32, 207], "models": ["gru"], "sensors": 16}
    path = tmp_path / "c.json"
    path.write_text(json.dumps(good))
    counts, config = scaling.load_config(path)
    assert counts == [16, 32, 207] and config.models == ["gru"]
    for bad in ([], [32, 16], [8, 16], [16, 16], [16, 300]):
        path.write_text(json.dumps({**good, "sensor_counts": bad}))
        with pytest.raises(ValueError):
            scaling.load_config(path)


def test_summary_separates_own_population_from_the_common_core(tmp_path):
    write_size(tmp_path / "n016", 16)
    write_size(tmp_path / "n064", 64)
    write_size(tmp_path / "n032", 32, status="running")  # unfinished sizes are skipped
    summary = scaling.summarize(tmp_path)
    assert summary["sizes"] == [16, 64]
    big = summary["models"]["sgru_time"]["64"]
    small = summary["models"]["sgru_time"]["16"]
    assert (
        small["core16_mae"]["mean"] == small["mae"]["mean"]
    )  # 16 sensors: same targets
    assert big["core16_mae"]["mean"] != big["mae"]["mean"]
    assert big["core16_mae"]["mean"] == pytest.approx(3.2 + 0.01 * 4)
    assert big["mae"]["n"] == 2 and big["mae"]["sd"] is not None
    assert summary["models"]["ha"]["64"]["mae"]["sd"] is None  # a single static run
    assert summary["models"]["ha"]["16"]["parameters"] == 0
    assert (
        big["epochs_run"]["mean"] == 4
        and summary["models"]["ha"]["16"]["epochs_run"] is None
    )
    assert big["peak_memory_mib"]["mean"] == pytest.approx(300)
    assert "not comparable" in summary["note"]
    assert (tmp_path / "scaling-summary.json").exists()


def test_figures_are_written_from_a_summary(tmp_path):
    write_size(tmp_path / "n016", 16)
    write_size(tmp_path / "n032", 32)
    scaling.figures(scaling.summarize(tmp_path), tmp_path)
    for name in ("scaling-accuracy.png", "scaling-cost.png"):
        assert (tmp_path / name).stat().st_size > 0


def test_finished_sizes_are_skipped_and_unfinished_ones_set_aside(
    tmp_path, monkeypatch
):
    write_size(tmp_path / "n016", 16)
    write_size(tmp_path / "n032", 32, status="running")
    ran = []

    def fake_benchmark(config, data, graph, folder):
        ran.append((config.sensors, config.core_sizes))
        write_size(folder, config.sensors)

    monkeypatch.setattr(scaling, "run_benchmark", fake_benchmark)
    counts, config = [16, 32, 64], scaling.BenchmarkConfig(models=["gru"])
    scaling.run(counts, config, "unused", "unused", tmp_path)
    assert ran == [(32, [16]), (64, [16, 32])]  # 16 was already done
    assert any(p.name.startswith("n032.unfinished") for p in tmp_path.iterdir())
    assert (
        json.loads((tmp_path / "n032" / "status.json").read_text())["status"]
        == "complete"
    )
