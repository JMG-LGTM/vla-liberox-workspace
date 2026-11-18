"""stop_after_success: strict API, drafts, inheritance, and saved provenance."""
from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from fastapi import FastAPI, Request
from pydantic import ValidationError

from backend.app.api.drafts import create_draft, update_draft
from backend.app.api.models import DraftRequest, UpdateDraftRequest
from backend.app.api.runs import create_session
from backend.app.storage.files import scan_trajectories
from backend.app.workers.simulation_worker import SimulationManager
from test_ui_tasks_draft import _DraftCatalog, _PolicyCatalog
from trajectory_utils import TrajectoryRecorder, save_trajectory_bundle


def manager_fixture(tmp_path):
    manager = object.__new__(SimulationManager)
    manager.lock = threading.RLock()
    manager.active_session_id = None
    manager.draft = None
    manager.catalog = _DraftCatalog()
    manager.catalog.paths = lambda _: (tmp_path / "task.bddl", tmp_path / "task.init")
    manager.policy_catalog = _PolicyCatalog()
    manager.eval_config = SimpleNamespace(seed=0, disabled_policy_cameras=(), control_hz=20)
    manager.ui_config = SimpleNamespace(output_root=tmp_path)
    manager.provider = SimpleNamespace(metadata=lambda: {"model_device": "cpu"})
    manager._launch_draft_preview = lambda draft: (
        setattr(draft, "latest_jpeg", b"jpeg"), setattr(draft, "preview_status", "READY")
    )
    return manager


def test_option_api_defaults_are_false_and_updates_can_explicitly_disable():
    request = DraftRequest(task_id="task", max_steps=50)
    assert request.stop_after_success is False
    for field in ("stop_after_success",):
        assert getattr(UpdateDraftRequest(**{field: False}), field) is False
        for value in (1, 0, "true", "false"):
            with pytest.raises(ValidationError):
                DraftRequest(task_id="task", max_steps=50, **{field: value})
            with pytest.raises(ValidationError):
                UpdateDraftRequest(**{field: value})
    with pytest.raises(ValidationError):
        UpdateDraftRequest()


def test_draft_api_and_direct_sessions_forward_options():
    calls = []
    app = FastAPI()
    app.state.manager = SimpleNamespace(
        create_draft=lambda *args, **kwargs: calls.append(("draft", kwargs)) or {},
        update_draft=lambda **kwargs: calls.append(("update", kwargs)) or {},
        create_original=lambda **kwargs: calls.append(("session", kwargs)) or {},
    )
    request = Request({"type": "http", "app": app})
    body = DraftRequest(task_id="task", max_steps=50, stop_after_success=True)
    asyncio.run(create_draft(body, request))
    asyncio.run(create_session(body, request))
    asyncio.run(update_draft(UpdateDraftRequest(stop_after_success=False), request))
    assert [(name, kwargs["stop_after_success"]) for name, kwargs in calls] == [
        ("draft", True), ("session", True), ("update", False),
    ]


def test_options_update_without_rebuilding_preview_and_start_carries_them(tmp_path):
    manager = manager_fixture(tmp_path)
    manager.create_draft("LEVEL1::task_a", 300)
    old_revision, old_frame = manager.draft.preview_revision, manager.draft.latest_jpeg
    manager._launch_draft_preview = lambda _: pytest.fail("Options must not rebuild the preview")
    enabled = manager.update_draft(stop_after_success=True)
    assert enabled["stop_after_success"]
    assert manager.draft.preview_revision == old_revision
    assert manager.draft.latest_jpeg == old_frame and manager.draft.preview_status == "READY"
    disabled = manager.update_draft(stop_after_success=False)
    assert disabled["stop_after_success"] is False
    manager.update_draft(stop_after_success=True)
    started = {}
    manager.create_original = lambda *args, **kwargs: started.update(kwargs) or {"id": "new"}
    manager.controller_status = lambda _: {"state": "READY"}
    assert manager.start_draft() == {"id": "new"}
    assert started["stop_after_success"] is True


@pytest.mark.parametrize("enabled", [False, True])
def test_record_options_persist_and_old_records_default_off(tmp_path, enabled):
    manager = manager_fixture(tmp_path)
    entry = manager.policy_catalog.entry
    def policy_entry(policy_id):
        result = entry(policy_id)
        result.family, result.stats_key = "fixture", None
        return result
    manager.policy_catalog.entry = policy_entry
    record = manager._new_record(kind="original", max_steps=100, open_loop_steps=8,
        stop_after_success=enabled)
    manager._persist_manifest(record)
    manifest = json.loads((record.output_dir / "run.json").read_text())
    config = yaml.safe_load((record.output_dir / "config.yaml").read_text())
    metadata = manager._metadata(record, {}, None)
    summary = manager._summary(record, {}, None, None)
    views = [record.public(), manifest, config["simulation"], metadata,
             summary, manager._public_from_persisted(record.output_dir, manifest, {})]
    for view in views:
        assert view["stop_after_success"] is enabled
    old_view = manager._public_from_persisted(tmp_path, {"id": "old"}, {})
    assert old_view["stop_after_success"] is False


@pytest.mark.parametrize("metadata_options", [{}, {"stop_after_success": True}])
def test_trajectory_index_preserves_options_without_altering_files(tmp_path, metadata_options):
    observation = {"robot0_eef_pos": np.zeros(3), "robot0_eef_quat": np.array([0, 0, 0, 1]),
                   "robot0_gripper_qpos": np.zeros(2)}
    env = SimpleNamespace(get_sim_state=lambda: np.zeros(1))
    recorder = TrajectoryRecorder(capture_images=False)
    recorder.record_initial(env, observation)
    recorder.record_transition(env, observation, np.zeros(7), np.zeros(7), 0., False, "policy")
    save_trajectory_bundle(recorder, tmp_path / "trajectory", {"control_hz": 20., **metadata_options},
                           create_plot=False)
    original = (tmp_path / "trajectory.npz").read_bytes()
    run = scan_trajectories([tmp_path])[0]
    assert run["stop_after_success"] is metadata_options.get("stop_after_success", False)
    assert (tmp_path / "trajectory.npz").read_bytes() == original


@pytest.mark.parametrize("options", [{}, {"stop_after_success": True}])
def test_new_branch_inherits_options_from_parent_and_old_parent_defaults(tmp_path, options):
    manager = manager_fixture(tmp_path)
    manager._persist_effective_config = lambda *args: None
    parent = {"id": "parent", "trajectory": "/tmp/parent.npz", "task_id": "LEVEL1::task_a",
              "policy_id": "base", "policy_content_sha256": "base-weights", **options}
    branch = manager._new_record(kind="branch", max_steps=100, open_loop_steps=8,
                                 parent=parent, resume_step=3)
    assert branch.stop_after_success is options.get("stop_after_success", False)


def test_create_branch_uses_saved_source_options(tmp_path, monkeypatch):
    manager = manager_fixture(tmp_path)
    parent = {"id": "parent", "trajectory": "/tmp/source.npz", "task_id": "LEVEL1::task_a",
              "policy_id": "base", "policy_content_sha256": "base-weights", "branchable": True,
              "status": "COMPLETED", "action_count": 10, "max_steps": 100,
              "stop_after_success": False}
    manager.get_public = lambda _: parent
    monkeypatch.setattr("backend.app.workers.simulation_worker.load_trajectory", lambda _: ({}, {
        "stop_after_success": True,
        "session_id": "parent",
    }))
    manager._persist_effective_config = lambda *args: None
    manager.work_queue = object()
    manager._start_record = lambda record: record.public()
    child = manager.create_branch("parent", 3, "policy", 8)
    assert child["stop_after_success"] is True
    assert parent["stop_after_success"] is False  # Reading saved inputs does not mutate history.
