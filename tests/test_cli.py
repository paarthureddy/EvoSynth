"""The `evosynth` command: parsing, settings commands, error handling (no training)."""
import json

import pytest

from evosynth.cli import build_parser, main


def test_settings_check(capsys):
    assert main(["settings", "check"]) == 0
    assert "settings OK" in capsys.readouterr().out


def test_settings_show_resolves_preset_and_overrides(capsys):
    assert main(["settings", "show", "engine", "--preset", "search_full", "--set", "engine.budget=600"]) == 0
    engine = json.loads(capsys.readouterr().out)
    assert engine["budget"] == 600 and engine["pop_size"] == 40


def test_invalid_setting_stops_before_running(capsys):
    # validation fails before any data is loaded or any search starts
    assert main(["run", "--set", "composite.fitness=0.9"]) == 2
    assert "settings error" in capsys.readouterr().err


def test_datasets_list(capsys):
    assert main(["datasets", "list"]) == 0
    assert "covertype" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["run", "--pop_size", "30"], ["run", "--pop-size", "30"]])
def test_old_and_new_flag_spellings(argv):
    assert build_parser().parse_args(argv).pop_size == 30


def test_every_command_has_help():
    parser = build_parser()
    for cmd in ["run", "scale", "ablation", "baselines", "nas", "serve", "settings", "datasets"]:
        with pytest.raises(SystemExit) as exc:
            parser.parse_args([cmd, "--help"])
        assert exc.value.code == 0
