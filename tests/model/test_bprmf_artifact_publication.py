import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

from Application.model import BPRMF


class _SyntheticModel:
    def __init__(self, generation="new"):
        self.generation = generation

    def state_dict(self):
        return {"generation": self.generation}


def _synthetic_mappings(generation="new"):
    return BPRMF.Mappings(
        user2idx={f"{generation}-user": 0},
        idx2user=[f"{generation}-user"],
        item2idx={f"{generation}-item": 0},
        idx2item=[f"{generation}-item"],
    )


def _checkpoint(generation, *, num_users=1, num_items=1):
    return {
        "model_type": "bprmf",
        "config": BPRMF.TrainConfig().__dict__,
        "num_users": num_users,
        "num_items": num_items,
        "state_dict": {"generation": generation},
        "feat2idx": {},
        "item_feat_mat": None,
        "item_feature_cols": [],
        "max_item_features": 32,
        "train_item_meta": {},
    }


def _write_generation(model_dir, generation, *, num_users=1, num_items=1):
    generation_dir = model_dir / "runs" / generation
    generation_dir.mkdir(parents=True)
    mappings = {
        "idx2user": [f"{generation}-user"] * num_users,
        "idx2item": [f"{generation}-item"] * num_items,
    }
    (generation_dir / "mappings.json").write_text(
        json.dumps(mappings), encoding="utf-8"
    )
    torch.save(
        _checkpoint(generation, num_users=num_users, num_items=num_items),
        generation_dir / "bprmf.pt",
    )
    return generation_dir


def _publish_manifest(model_dir, generation):
    manifest_path = model_dir / "current.json"
    manifest_path.write_text(
        json.dumps({"generation": generation}), encoding="utf-8"
    )
    return manifest_path


def _prepare_current_generation(tmp_path, generation="generation-a"):
    model_dir = tmp_path / "Модель"
    model_dir.mkdir()
    _write_generation(model_dir, generation)
    manifest_path = _publish_manifest(model_dir, generation)
    return model_dir, manifest_path, manifest_path.read_bytes()


@pytest.fixture
def lightweight_artifact_preparation(monkeypatch):
    monkeypatch.setattr(
        BPRMF,
        "_build_item_feature_matrix",
        lambda *args, **kwargs: ({}, None),
    )


def test_partial_mappings_write_keeps_legacy_model_unchanged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model_dir = tmp_path / "Модель"
    model_dir.mkdir()
    legacy_mappings = model_dir / "mappings.json"
    legacy_checkpoint = model_dir / "bprmf.pt"
    old_mappings_bytes = b'{"idx2user":["old-user"],"idx2item":["old-item"]}'
    old_checkpoint_bytes = b"synthetic-old-checkpoint"
    legacy_mappings.write_bytes(old_mappings_bytes)
    legacy_checkpoint.write_bytes(old_checkpoint_bytes)

    def partial_dump(data, file_object, **kwargs):
        file_object.write('{"partial"')
        raise OSError("synthetic mappings write error")

    monkeypatch.setattr(BPRMF.json, "dump", partial_dump)

    with pytest.raises(OSError, match="synthetic mappings write error"):
        BPRMF._save_artifacts(
            BPRMF.TrainConfig(data_dir=str(tmp_path / "synthetic-data")),
            _synthetic_mappings(),
            _SyntheticModel(),
        )

    assert legacy_mappings.read_bytes() == old_mappings_bytes
    assert legacy_checkpoint.read_bytes() == old_checkpoint_bytes
    assert not (model_dir / "current.json").exists()
    assert list((model_dir / "runs").iterdir()) == []
    assert list((model_dir / ".staging").iterdir()) == []


def test_checkpoint_preparation_error_keeps_current_generation(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _model_dir, manifest_path, old_manifest_bytes = _prepare_current_generation(
        tmp_path
    )

    def fail_checkpoint_preparation(*args, **kwargs):
        raise RuntimeError("synthetic checkpoint preparation error")

    monkeypatch.setattr(
        BPRMF,
        "_build_item_feature_matrix",
        fail_checkpoint_preparation,
    )

    with pytest.raises(RuntimeError, match="synthetic checkpoint preparation error"):
        BPRMF._save_artifacts(
            BPRMF.TrainConfig(data_dir=str(tmp_path / "synthetic-data")),
            _synthetic_mappings(),
            _SyntheticModel(),
        )

    assert manifest_path.read_bytes() == old_manifest_bytes
    mappings, checkpoint = BPRMF._load_artifacts()
    assert mappings["idx2user"] == ["generation-a-user"]
    assert checkpoint["state_dict"]["generation"] == "generation-a"
    assert list((_model_dir / ".staging").iterdir()) == []


def test_partial_torch_save_keeps_current_generation(
    tmp_path, monkeypatch, lightweight_artifact_preparation
):
    monkeypatch.chdir(tmp_path)
    _model_dir, manifest_path, old_manifest_bytes = _prepare_current_generation(
        tmp_path
    )

    def partial_save(checkpoint, destination):
        destination.write(b"partial-checkpoint")
        raise OSError("synthetic checkpoint write error")

    monkeypatch.setattr(BPRMF.torch, "save", partial_save)

    with pytest.raises(OSError, match="synthetic checkpoint write error"):
        BPRMF._save_artifacts(
            BPRMF.TrainConfig(data_dir=str(tmp_path / "synthetic-data")),
            _synthetic_mappings(),
            _SyntheticModel(),
        )

    assert manifest_path.read_bytes() == old_manifest_bytes
    mappings, checkpoint = BPRMF._load_artifacts()
    assert mappings["idx2user"] == ["generation-a-user"]
    assert checkpoint["state_dict"]["generation"] == "generation-a"
    assert list((_model_dir / ".staging").iterdir()) == []


def test_generation_rename_error_does_not_switch_current(
    tmp_path, monkeypatch, lightweight_artifact_preparation
):
    monkeypatch.chdir(tmp_path)
    _model_dir, manifest_path, old_manifest_bytes = _prepare_current_generation(
        tmp_path
    )

    def fail_rename(source, destination):
        raise OSError("synthetic generation rename error")

    monkeypatch.setattr(BPRMF.os, "rename", fail_rename)

    with pytest.raises(OSError, match="synthetic generation rename error"):
        BPRMF._save_artifacts(
            BPRMF.TrainConfig(data_dir=str(tmp_path / "synthetic-data")),
            _synthetic_mappings(),
            _SyntheticModel(),
        )

    assert manifest_path.read_bytes() == old_manifest_bytes
    mappings, checkpoint = BPRMF._load_artifacts()
    assert mappings["idx2user"] == ["generation-a-user"]
    assert checkpoint["state_dict"]["generation"] == "generation-a"
    assert list((_model_dir / ".staging").iterdir()) == []


def test_manifest_replace_error_keeps_old_manifest_and_generation(
    tmp_path, monkeypatch, lightweight_artifact_preparation
):
    monkeypatch.chdir(tmp_path)
    model_dir, manifest_path, old_manifest_bytes = _prepare_current_generation(
        tmp_path
    )
    other_publisher_temp = model_dir / ".current.other-publisher.tmp"
    other_publisher_bytes = b'{"generation":"other-publisher"}'
    other_publisher_temp.write_bytes(other_publisher_bytes)
    failed_manifest_temps = []

    def fail_replace(source, destination):
        assert Path(destination) == manifest_path
        failed_manifest_temps.append(Path(source))
        raise OSError("synthetic manifest publish error")

    monkeypatch.setattr(BPRMF.os, "replace", fail_replace)

    with pytest.raises(OSError, match="synthetic manifest publish error"):
        BPRMF._save_artifacts(
            BPRMF.TrainConfig(data_dir=str(tmp_path / "synthetic-data")),
            _synthetic_mappings(),
            _SyntheticModel(),
        )

    assert manifest_path.read_bytes() == old_manifest_bytes
    mappings, checkpoint = BPRMF._load_artifacts()
    assert mappings["idx2user"] == ["generation-a-user"]
    assert checkpoint["state_dict"]["generation"] == "generation-a"
    assert len(list((model_dir / "runs").iterdir())) == 2
    assert len(failed_manifest_temps) == 1
    failed_manifest_temp = failed_manifest_temps[0]
    assert failed_manifest_temp.name.startswith(".current.")
    assert failed_manifest_temp.name.endswith(".tmp")
    assert not failed_manifest_temp.exists()
    assert other_publisher_temp.read_bytes() == other_publisher_bytes


def test_cleanup_error_does_not_hide_checkpoint_preparation_error(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    _model_dir, manifest_path, old_manifest_bytes = _prepare_current_generation(
        tmp_path
    )

    def fail_checkpoint_preparation(*args, **kwargs):
        raise RuntimeError("synthetic primary preparation error")

    def fail_cleanup(path):
        raise OSError("synthetic staging cleanup error")

    monkeypatch.setattr(
        BPRMF,
        "_build_item_feature_matrix",
        fail_checkpoint_preparation,
    )
    monkeypatch.setattr(BPRMF.shutil, "rmtree", fail_cleanup)

    with pytest.raises(RuntimeError, match="synthetic primary preparation error"):
        BPRMF._save_artifacts(
            BPRMF.TrainConfig(data_dir=str(tmp_path / "synthetic-data")),
            _synthetic_mappings(),
            _SyntheticModel(),
        )

    assert "synthetic staging cleanup error" in capsys.readouterr().err
    assert manifest_path.read_bytes() == old_manifest_bytes
    mappings, checkpoint = BPRMF._load_artifacts()
    assert mappings["idx2user"] == ["generation-a-user"]
    assert checkpoint["state_dict"]["generation"] == "generation-a"


def test_successful_save_atomically_switches_to_one_new_generation(
    tmp_path, monkeypatch, lightweight_artifact_preparation
):
    monkeypatch.chdir(tmp_path)
    model_dir, _manifest_path, _old_manifest_bytes = _prepare_current_generation(
        tmp_path
    )

    BPRMF._save_artifacts(
        BPRMF.TrainConfig(data_dir=str(tmp_path / "synthetic-data")),
        _synthetic_mappings("generation-b"),
        _SyntheticModel("generation-b"),
    )

    manifest = json.loads((model_dir / "current.json").read_text(encoding="utf-8"))
    assert manifest["generation"] != "generation-a"
    current_dir = model_dir / "runs" / manifest["generation"]
    assert current_dir.is_dir()
    mappings, checkpoint = BPRMF._load_artifacts()
    assert mappings["idx2user"] == ["generation-b-user"]
    assert mappings["idx2item"] == ["generation-b-item"]
    assert checkpoint["state_dict"]["generation"] == "generation-b"


def test_parallel_publishers_use_independent_manifest_temp_files(
    tmp_path, monkeypatch, lightweight_artifact_preparation
):
    monkeypatch.chdir(tmp_path)
    model_dir, _manifest_path, _old_manifest_bytes = _prepare_current_generation(
        tmp_path
    )
    real_replace = BPRMF.os.replace
    publication_barrier = threading.Barrier(2)
    recorded_manifests = []
    records_lock = threading.Lock()
    replace_lock = threading.Lock()

    def synchronized_replace(source, destination):
        source_path = Path(source)
        manifest = json.loads(source_path.read_text(encoding="utf-8"))
        with records_lock:
            recorded_manifests.append((source_path, manifest["generation"]))
        publication_barrier.wait(timeout=10)
        with replace_lock:
            real_replace(source, destination)

    monkeypatch.setattr(BPRMF.os, "replace", synchronized_replace)

    def publish(label):
        try:
            BPRMF._save_artifacts(
                BPRMF.TrainConfig(data_dir=str(tmp_path / "synthetic-data")),
                _synthetic_mappings(label),
                _SyntheticModel(label),
            )
        except Exception as error:
            return error
        return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        errors = list(executor.map(publish, ["publisher-a", "publisher-b"]))

    assert errors == [None, None]
    assert len(recorded_manifests) == 2
    assert len({path for path, _generation in recorded_manifests}) == 2
    for path, generation in recorded_manifests:
        assert path.name == f".current.{generation}.tmp"
        assert not path.exists()

    current = json.loads((model_dir / "current.json").read_text(encoding="utf-8"))
    published_generations = {
        generation for _path, generation in recorded_manifests
    }
    assert current["generation"] in published_generations
    current_dir = model_dir / "runs" / current["generation"]
    assert (current_dir / "mappings.json").is_file()
    assert (current_dir / "bprmf.pt").is_file()
    mappings, checkpoint = BPRMF._load_artifacts()
    mapping_label = mappings["idx2user"][0].removesuffix("-user")
    assert checkpoint["state_dict"]["generation"] == mapping_label


def test_reader_keeps_selected_generation_when_manifest_switches_mid_load(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    model_dir = tmp_path / "Модель"
    model_dir.mkdir()
    _write_generation(model_dir, "generation-a")
    _write_generation(model_dir, "generation-b")
    _publish_manifest(model_dir, "generation-a")
    real_json_load = json.load

    def load_and_switch_manifest(file_object, *args, **kwargs):
        result = real_json_load(file_object, *args, **kwargs)
        if Path(file_object.name).name == "mappings.json":
            _publish_manifest(model_dir, "generation-b")
        return result

    monkeypatch.setattr(BPRMF.json, "load", load_and_switch_manifest)

    mappings, checkpoint = BPRMF._load_artifacts()

    assert mappings["idx2user"] == ["generation-a-user"]
    assert checkpoint["state_dict"]["generation"] == "generation-a"
    assert json.loads(
        (model_dir / "current.json").read_text(encoding="utf-8")
    )["generation"] == "generation-b"


def test_loader_falls_back_to_legacy_pair_when_manifest_is_absent(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    model_dir = tmp_path / "Модель"
    model_dir.mkdir()
    legacy_mappings = {
        "idx2user": ["legacy-user"],
        "idx2item": ["legacy-item"],
    }
    (model_dir / "mappings.json").write_text(
        json.dumps(legacy_mappings), encoding="utf-8"
    )
    torch.save(_checkpoint("legacy"), model_dir / "bprmf.pt")

    mappings, checkpoint = BPRMF._load_artifacts()

    assert mappings == legacy_mappings
    assert checkpoint["state_dict"]["generation"] == "legacy"


def test_first_successful_generation_preserves_legacy_files(
    tmp_path, monkeypatch, lightweight_artifact_preparation
):
    monkeypatch.chdir(tmp_path)
    model_dir = tmp_path / "Модель"
    model_dir.mkdir()
    legacy_mappings_path = model_dir / "mappings.json"
    legacy_checkpoint_path = model_dir / "bprmf.pt"
    legacy_mappings_path.write_text(
        json.dumps(
            {"idx2user": ["legacy-user"], "idx2item": ["legacy-item"]}
        ),
        encoding="utf-8",
    )
    torch.save(_checkpoint("legacy"), legacy_checkpoint_path)
    legacy_mappings_bytes = legacy_mappings_path.read_bytes()
    legacy_checkpoint_bytes = legacy_checkpoint_path.read_bytes()

    BPRMF._save_artifacts(
        BPRMF.TrainConfig(data_dir=str(tmp_path / "synthetic-data")),
        _synthetic_mappings("generation-b"),
        _SyntheticModel("generation-b"),
    )

    assert legacy_mappings_path.read_bytes() == legacy_mappings_bytes
    assert legacy_checkpoint_path.read_bytes() == legacy_checkpoint_bytes
    assert (model_dir / "current.json").is_file()
    mappings, checkpoint = BPRMF._load_artifacts()
    assert mappings["idx2user"] == ["generation-b-user"]
    assert checkpoint["state_dict"]["generation"] == "generation-b"


def test_export_uses_shared_generation_aware_loader(monkeypatch):
    def fail_after_shared_loader(model_dir):
        assert model_dir == "synthetic-model-dir"
        raise RuntimeError("synthetic shared loader marker")

    monkeypatch.setattr(BPRMF, "_load_artifacts", fail_after_shared_loader)

    with pytest.raises(RuntimeError, match="synthetic shared loader marker"):
        BPRMF.export_recommendations_excel(model_dir="synthetic-model-dir")


@pytest.mark.parametrize(
    ("num_users", "num_items", "expected_field"),
    [(2, 1, "idx2user"), (1, 2, "idx2item")],
)
def test_loader_rejects_inconsistent_current_generation_without_legacy_fallback(
    tmp_path, monkeypatch, num_users, num_items, expected_field
):
    monkeypatch.chdir(tmp_path)
    model_dir = tmp_path / "Модель"
    model_dir.mkdir()
    _write_generation(
        model_dir,
        "broken-generation",
        num_users=num_users,
        num_items=num_items,
    )
    generation_dir = model_dir / "runs" / "broken-generation"
    (generation_dir / "mappings.json").write_text(
        json.dumps(
            {
                "idx2user": ["one-user"],
                "idx2item": ["one-item"],
            }
        ),
        encoding="utf-8",
    )
    _publish_manifest(model_dir, "broken-generation")
    (model_dir / "mappings.json").write_text(
        json.dumps(
            {"idx2user": ["legacy-user"], "idx2item": ["legacy-item"]}
        ),
        encoding="utf-8",
    )
    torch.save(_checkpoint("legacy"), model_dir / "bprmf.pt")

    with pytest.raises(ValueError, match=expected_field):
        BPRMF._load_artifacts()
