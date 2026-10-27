"""Installed OmniVoice weights must load without resolving a remote repo name."""
import io
import socket
import threading
import types

import pytest


class LoadReached(RuntimeError):
    """Stop at the expensive model-construction boundary."""


@pytest.mark.parametrize("runtime", ["inprocess", "sidecar"])
@pytest.mark.parametrize("source_kind", ["installed", "missing", "directory"])
def test_omnivoice_load_source(tmp_path, monkeypatch, runtime, source_kind):
    from services import model_manager as mm
    from engines.omnivoice_subprocess import main as sidecar
    from services.hf_revisions import revision_for

    repo = "k2-fsa/OmniVoice"
    cache = tmp_path / "hub"
    cache.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    source = expected = repo
    if source_kind == "installed":
        repo_dir = cache / "models--k2-fsa--OmniVoice"
        snapshot = repo_dir / "snapshots" / revision_for(repo)
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}")
        with (snapshot / "model.safetensors").open("wb") as weights:
            weights.truncate(64 * 1024 * 1024)
        # Pinned installs need not have refs/main.
        (repo_dir / "voicestudio-revision").write_text(revision_for(repo) + "\n")
        expected = str(snapshot)
    elif source_kind == "directory":
        local = tmp_path / "custom-model"
        local.mkdir()
        source = expected = str(local)
    monkeypatch.setenv("OMNIVOICE_MODEL", source)

    def refuse_network(*_args, **_kwargs):
        raise AssertionError("network access while resolving an installed model")

    monkeypatch.setattr(socket.socket, "connect", refuse_network)
    monkeypatch.setattr(socket, "getaddrinfo", refuse_network)

    class FakeModel:
        @staticmethod
        def from_pretrained(checkpoint, **_kwargs):
            assert checkpoint == expected
            raise LoadReached("model constructor reached")

    monkeypatch.setattr(mm, "_lazy_omnivoice", lambda: FakeModel)
    monkeypatch.setattr(mm, "_lazy_torch", lambda: types.SimpleNamespace(float32="float32"))
    monkeypatch.setattr(mm, "get_best_device", lambda: "cpu")
    monkeypatch.setattr(mm, "should_preload_tts_asr", lambda: False)
    monkeypatch.setattr(mm, "_preflight_truncated_weights", lambda _source: None)
    monkeypatch.setattr(mm, "_set_loading", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mm, "_shutting_down", threading.Event())
    monkeypatch.setattr(sidecar, "_model", None)
    monkeypatch.setattr(sidecar, "_prefetch_weights", lambda *_args: None)

    with pytest.raises(LoadReached, match="model constructor reached"):
        if runtime == "sidecar":
            sidecar._load_model(io.BytesIO())
        else:
            mm._load_model_sync()
