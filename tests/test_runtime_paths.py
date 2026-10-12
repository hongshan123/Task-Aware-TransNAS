from scripts.runtime_paths import data_root, resolve_config_paths, resolve_runtime_path, storage_root


def test_storage_root_defaults_to_project_root_on_linux(monkeypatch, tmp_path):
    monkeypatch.delenv("TRANSNAS_STORAGE_ROOT", raising=False)

    assert storage_root(tmp_path, platform="linux") == (tmp_path / "runs").resolve()


def test_relative_storage_override_is_relative_to_project_root(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSNAS_STORAGE_ROOT", "server-data")

    assert storage_root(tmp_path, platform="linux") == (tmp_path / "server-data").resolve()


def test_legacy_windows_storage_path_maps_to_linux_storage_root(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSNAS_STORAGE_ROOT", str(tmp_path / "runs"))

    resolved = resolve_runtime_path(
        r"D:\Task-Aware-TransNAS\results\phase4b\metrics.json",
        project_root=tmp_path,
    )

    assert resolved == tmp_path / "runs" / "results" / "phase4b" / "metrics.json"


def test_legacy_windows_data_path_maps_to_project_data_root(monkeypatch, tmp_path):
    monkeypatch.delenv("TRANSNAS_STORAGE_ROOT", raising=False)
    monkeypatch.delenv("TRANSNAS_DATA_ROOT", raising=False)

    assert resolve_runtime_path(
        r"D:\Task-Aware-TransNAS\data\CITB-official\data\tasks.json",
        project_root=tmp_path,
    ) == tmp_path / "data" / "CITB-official" / "data" / "tasks.json"


def test_data_root_accepts_environment_override(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSNAS_DATA_ROOT", "datasets")

    assert data_root(tmp_path) == (tmp_path / "datasets").resolve()


def test_resolve_config_paths_normalizes_dataset_and_storage_paths(monkeypatch, tmp_path):
    monkeypatch.delenv("TRANSNAS_STORAGE_ROOT", raising=False)
    monkeypatch.delenv("TRANSNAS_DATA_ROOT", raising=False)
    config = {
        "dataset": {
            "source_repo": r"D:\Task-Aware-TransNAS\data\CITB-official",
            "source_data": r"D:\Task-Aware-TransNAS\data\CITB-official\data",
            "processed_dir": r"D:\Task-Aware-TransNAS\data\processed\citb_formal_4tasks",
        },
        "storage": {
            "root": r"D:\Task-Aware-TransNAS",
            "checkpoints": r"D:\Task-Aware-TransNAS\checkpoints\baseline_full",
            "results": r"D:\Task-Aware-TransNAS\results\baseline_full",
        },
    }

    resolved = resolve_config_paths(config, project_root=tmp_path, platform="linux")

    assert resolved["dataset"]["source_repo"] == str(tmp_path / "data" / "CITB-official")
    assert resolved["dataset"]["source_data"] == str(tmp_path / "data" / "CITB-official" / "data")
    assert resolved["dataset"]["processed_dir"] == str(tmp_path / "data" / "processed" / "citb_formal_4tasks")
    assert resolved["storage"]["root"] == str(tmp_path / "runs")
    assert resolved["storage"]["results"] == str(tmp_path / "runs" / "results" / "baseline_full")


def test_project_relative_runtime_path_is_rooted_at_project(monkeypatch, tmp_path):
    monkeypatch.delenv("TRANSNAS_STORAGE_ROOT", raising=False)

    assert resolve_runtime_path("data/processed/train.jsonl", project_root=tmp_path) == (
        tmp_path / "data" / "processed" / "train.jsonl"
    )
