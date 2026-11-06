"""The arms x seeds runner and its summary, on constructed data."""

import json

import pytest

from tracast import fedgraph_study as S

from test_fedgraph import GROUPS, make_prep, synthetic


def settings(**base):
    return {
        "name": "constructed",
        "base": {
            "sensors": 16,
            "rows": 1500,
            "hidden": 4,
            "clients": 3,
            "rounds": 2,
            "local_stride": 2,
            "batch_size": 16,
            "threads": 1,
            "model": "sgraph",
            **base,
        },
        "seeds": [1, 2],
        "arms": [
            {"name": "fed", "mode": "federated", "runner": "inprocess"},
            {
                "name": "fed_prox",
                "mode": "federated",
                "algorithm": "fedprox",
                "proximal_mu": 0.1,
            },
            {"name": "central_full", "mode": "central", "edges": "full"},
            {"name": "central_local", "mode": "central", "edges": "local"},
            {"name": "local", "mode": "local"},
            {"name": "plain", "mode": "federated", "model": "sgru"},
        ],
        "pairs": [["fed", "plain"], ["fed", "central_local"], ["fed", "ha"]],
    }


@pytest.fixture
def constructed(monkeypatch):
    frame, weights = synthetic(rows=1500)
    prep = make_prep(frame, weights, GROUPS)
    monkeypatch.setattr(S.F, "prepare", lambda config, data, graph: prep)
    return prep


def write(tmp_path, content):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(content))
    return path


# --- settings -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "edit",
    [
        lambda s: s["arms"].append({"name": "fed", "mode": "local"}),  # duplicate name
        lambda s: s["arms"].append({"name": "x", "mode": "gossip"}),
        lambda s: s["arms"].append({"name": "x", "mode": "local", "runner": "mpi"}),
        lambda s: s["arms"].append(
            {"name": "x", "mode": "central", "runner": "flower"}
        ),
        lambda s: s["arms"].append({"name": "x", "mode": "central", "edges": "all"}),
        lambda s: s["arms"].append({"name": "x", "mode": "local", "optimizer": "sgd"}),
        lambda s: s["arms"].append({"name": "x", "mode": "federated", "rounds": 0}),
        lambda s: s.update(seeds=[1, 1]),
        lambda s: s.update(seeds=[]),
        lambda s: s.update(pairs=[["fed", "nope"]]),
        lambda s: s["arms"][0].update(seeds=[1]),  # a pair needs matching seeds
    ],
)
def test_settings_reject_invalid_studies(tmp_path, edit):
    content = settings()
    edit(content)
    with pytest.raises(ValueError):
        S.load_settings(write(tmp_path, content))


def test_an_arm_can_use_fewer_seeds_when_it_is_not_paired(tmp_path):
    content = settings()
    content["arms"].append({"name": "cheap", "mode": "local", "seeds": [1]})
    loaded = S.load_settings(write(tmp_path, content))
    assert S.arm_seeds(loaded, S.find_arm(loaded, "cheap")) == [1]
    assert S.arm_seeds(loaded, S.find_arm(loaded, "fed")) == [1, 2]


def test_arm_overrides_change_only_the_named_fields():
    study = settings()
    config = S.arm_config(study, S.find_arm(study, "fed_prox"), 5)
    assert config.algorithm == "fedprox" and config.proximal_mu == 0.1
    assert config.seed == 5 and config.clients == 3 and config.rounds == 2
    assert S.arm_config(study, S.find_arm(study, "fed"), 7).algorithm == "fedavg"


def test_shipped_configs_are_valid():
    for name in ("fedgraph", "fedgraph-diffusion", "fedgraph-optimization"):
        loaded = S.load_settings(f"configs/{name}.json")
        assert loaded["arms"] and loaded["pairs"]


# --- cases ----------------------------------------------------------------------------


def test_every_arm_mode_writes_a_complete_case_folder(tmp_path, constructed):
    study = settings()
    results = {}
    for name in ("fed", "fed_prox", "central_full", "central_local", "local", "plain"):
        results[name] = S.run_case(study, name, 1, "data", "graph", tmp_path)
        folder = tmp_path / f"{name}-s1"
        assert json.loads((folder / "status.json").read_text())["status"] == "complete"
        for file in ("result.json", "setup.json", "manifest.json"):
            assert (folder / file).exists()
        result = results[name]
        assert result["arm"] == name and result["seed"] == 1
        assert result["test"]["mae"] > 0 and len(result["clients"]) == 3
        assert result["peak_memory"]["bytes"] > 0
    # all arms scored the same observed test targets
    counts = {n: r["test"]["per_horizon"][0]["count"] for n, r in results.items()}
    assert len(set(counts.values())) == 1
    manifest = json.loads((tmp_path / "fed-s1" / "manifest.json").read_text())
    assert manifest["versions"]["torch"] and manifest["code_sha256"]
    assert manifest["deterministic"]["mkldnn"] is False


def test_a_finished_case_is_not_run_again(tmp_path, constructed, monkeypatch):
    study = settings()
    first = S.run_case(study, "local", 1, "data", "graph", tmp_path)
    monkeypatch.setattr(S.F, "run_local_only", lambda *a, **k: pytest.fail("reran"))
    assert S.run_case(study, "local", 1, "data", "graph", tmp_path) == first


def test_an_unfinished_folder_is_set_aside_not_overwritten(tmp_path, constructed):
    folder = tmp_path / "local-s1"
    folder.mkdir()
    (folder / "partial.txt").write_text("half done")
    S.run_case(settings(), "local", 1, "data", "graph", tmp_path)
    leftovers = list(tmp_path.glob("local-s1.unfinished-*"))
    assert len(leftovers) == 1 and (leftovers[0] / "partial.txt").exists()


def test_the_pooled_arms_see_the_same_graph_as_the_federated_clients(
    tmp_path, constructed
):
    study = settings()
    full = S.run_case(study, "central_full", 1, "data", "graph", tmp_path)
    local = S.run_case(study, "central_local", 1, "data", "graph", tmp_path)
    assert (
        full["state_sha256"] != local["state_sha256"]
    )  # the cut graph changes training
    setup = json.loads((tmp_path / "central_local-s1" / "setup.json").read_text())
    kept = setup["graph"]["edges_kept"]
    assert kept == sum(c["local_edges"] for c in setup["clients"])


# --- the summary ----------------------------------------------------------------------


def test_summary_collects_arms_baselines_and_paired_comparisons(tmp_path, constructed):
    path = write(tmp_path, settings())
    out = tmp_path / "out"
    out.mkdir()
    study = S.load_settings(path)
    S.run_baselines(study, "data", "graph", out)
    for seed in (1, 2):
        for name in ("fed", "central_local", "plain"):
            S.run_case(study, name, seed, "data", "graph", out)
    summary = S.summarize(path, out)
    assert set(summary["arms"]) == {"fed", "central_local", "plain"}
    fed = summary["arms"]["fed"]
    assert fed["seeds"] == [1, 2] and fed["mae"]["n"] == 2 and fed["mae"]["sd"] >= 0
    assert fed["parameters"] > 0
    assert len(fed["horizon_mae"]) == 12 and len(fed["clients"]) == 3
    assert fed["communication"]["total_parameter_bytes"] > 0
    assert len(fed["validation_curve"]) == 2
    assert "communication" not in summary["arms"]["central_local"]
    assert {"persistence", "ha"} <= set(summary["baselines"])
    assert summary["partition"]["graph"]["directed_edges"] > 0
    gaps = summary["paired_day_bootstrap"]
    assert set(gaps) == {"fed_vs_plain", "fed_vs_central_local", "fed_vs_ha"}
    for gap in gaps.values():
        assert gap["ci95"][0] <= gap["estimate"] <= gap["ci95"][1] or gap["days"] < 5
    for figure in ("accuracy", "convergence", "horizons", "clients"):
        assert (out / f"{figure}.png").stat().st_size > 1000


def test_summary_works_with_only_some_arms_finished(tmp_path, constructed):
    path = write(tmp_path, settings())
    out = tmp_path / "out"
    out.mkdir()
    study = S.load_settings(path)
    S.run_baselines(study, "data", "graph", out)
    S.run_case(study, "local", 1, "data", "graph", out)
    summary = S.summarize(path, out)
    assert list(summary["arms"]) == ["local"] and summary["paired_day_bootstrap"] == {}


def test_budget_rows_are_summarized(tmp_path, constructed):
    content = settings(budgets=[1, 2])
    path = write(tmp_path, content)
    out = tmp_path / "out"
    out.mkdir()
    study = S.load_settings(path)
    S.run_baselines(study, "data", "graph", out)
    for seed in (1, 2):
        S.run_case(study, "fed", seed, "data", "graph", out)
    summary = S.summarize(path, out)
    table = summary["arms"]["fed"]["budgets"]
    assert set(table) == {"1", "2"}
    assert table["2"]["mae"]["mean"] == pytest.approx(
        summary["arms"]["fed"]["mae"]["mean"]
    )
    assert table["1"]["communication_bytes"] < table["2"]["communication_bytes"]


def test_a_fine_tuned_arm_gets_its_own_summary_row_and_pairs(tmp_path, constructed):
    content = settings()
    content["arms"] = [
        {
            "name": "fed",
            "mode": "federated",
            "runner": "inprocess",
            "personalize_rounds": 1,
        },
        {"name": "local", "mode": "local"},
    ]
    content["pairs"] = [["fed+ft", "local"], ["fed+ft", "fed"], ["fed+ft", "ha"]]
    path = write(tmp_path, content)
    out = tmp_path / "out"
    out.mkdir()
    study = S.load_settings(path)
    S.run_baselines(study, "data", "graph", out)
    for seed in (1, 2):
        for name in ("fed", "local"):
            S.run_case(study, name, seed, "data", "graph", out)
    summary = S.summarize(path, out)
    assert set(summary["arms"]) == {"fed", "fed+ft", "local"}
    tuned = summary["arms"]["fed+ft"]
    assert tuned["mode"] == "federated+finetune" and len(tuned["clients"]) == 3
    assert tuned["communication"] == summary["arms"]["fed"]["communication"]
    assert set(summary["paired_day_bootstrap"]) == {
        "fed+ft_vs_local",
        "fed+ft_vs_fed",
        "fed+ft_vs_ha",
    }
    assert all(c["fine_tune_round"]["mean"] >= 0 for c in tuned["clients"])


def test_a_pair_may_only_name_a_fine_tuned_arm_that_exists(tmp_path):
    content = settings()
    content["pairs"] = [["fed+ft", "local"]]  # fed has no fine-tuning rounds
    with pytest.raises(ValueError):
        S.load_settings(write(tmp_path, content))
