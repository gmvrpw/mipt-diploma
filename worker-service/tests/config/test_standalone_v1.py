from pathlib import Path

import pytest
import yaml

from src.config import ConfigError, parse_standalone_config


def _write(tmp_path: Path, data: dict) -> str:
    p = tmp_path / "config.yml"
    p.write_text(yaml.safe_dump(data))
    return str(p)


def test_defaults_when_empty(tmp_path: Path) -> None:
    path = _write(tmp_path, {"version": "v1"})
    config = parse_standalone_config(path, base_path=tmp_path)

    assert config.version == "v1"
    assert config.capabilities.add.hf_path == "Qwen/Qwen-Image-Edit-2509"
    assert config.capabilities.rmbg.hf_path == "plemeri/InSPyReNet"
    assert config.models.dir == str(tmp_path / "models")
    assert config.outputs.dir == str(tmp_path / "outputs")
    assert int(config.device.gpu_mem) == 80 * 1024 ** 3


def test_device_gpu_mem_parses_human_size(tmp_path: Path) -> None:
    path = _write(tmp_path, {
        "version": "v1",
        "device": {"gpu_mem": "40GiB"},
    })
    config = parse_standalone_config(path, base_path=tmp_path)
    assert int(config.device.gpu_mem) == 40 * 1024 ** 3


def test_absolute_path_kept(tmp_path: Path) -> None:
    abs_dir = "/var/lib/worker-models"
    path = _write(tmp_path, {
        "version": "v1",
        "models": {"dir": abs_dir},
    })
    config = parse_standalone_config(path, base_path=tmp_path)
    assert config.models.dir == abs_dir


def test_retries_int_expands_to_read_write(tmp_path: Path) -> None:
    path = _write(tmp_path, {
        "version": "v1",
        "outputs": {"retries": 5, "timeout": "200ms"},
    })
    config = parse_standalone_config(path, base_path=tmp_path)
    assert config.outputs.retries.read == 5
    assert config.outputs.retries.write == 5
    assert abs(config.outputs.timeout.read - 0.2) < 1e-9
    assert abs(config.outputs.timeout.write - 0.2) < 1e-9


def test_unknown_field_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, {
        "version": "v1",
        "models": {"dir": "models", "bogus": True},
    })
    with pytest.raises(ConfigError) as exc:
        parse_standalone_config(path, base_path=tmp_path)
    assert "models.bogus" in str(exc.value) or "Extra inputs" in str(exc.value)


def test_unsupported_version_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, {"version": "v999"})
    with pytest.raises(ConfigError) as exc:
        parse_standalone_config(path, base_path=tmp_path)
    assert "unsupported config version" in str(exc.value)
