"""The umbrella command forwards to each experiment's own parser."""

import json

import pytest

from tracast import cli


@pytest.mark.parametrize(
    "command,module",
    [
        ("benchmark", "tracast.benchmark"),
        ("scaling", "tracast.scaling"),
        ("uncertainty", "tracast.uncertainty"),
        ("shift", "tracast.shift"),
        ("fedgraph", "tracast.fedgraph_study"),
        ("holdout", "tracast.inductive"),
        ("federated", "tracast.federated"),
        ("study", "tracast.federated_study"),
        ("compare", "tracast.compare_runs"),
    ],
)
def test_delegated_commands_forward_remaining_arguments(command, module, monkeypatch):
    import importlib

    seen = []
    monkeypatch.setattr(
        importlib.import_module(module), "main", lambda argv: seen.append(argv)
    )
    cli.main([command, "--out", "somewhere", "--flag"])
    assert seen == [["--out", "somewhere", "--flag"]]


def test_config_prints_the_default_configuration(capsys):
    cli.main(["config"])
    assert json.loads(capsys.readouterr().out)["sensors"] == 16


def test_help_lists_every_command(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    text = capsys.readouterr().out
    for name in ("fetch", "inspect", "run", "config", *cli.DELEGATED):
        assert name in text


def test_unknown_command_is_rejected():
    with pytest.raises(SystemExit):
        cli.main(["no-such-command"])
