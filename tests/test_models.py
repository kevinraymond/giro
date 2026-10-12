"""giro models: the manifest covers every weight the workflows name; fetch fills the store."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from giro import models
from giro.comfy import server

ROOT = Path(__file__).resolve().parents[1] / "src" / "giro"
PAT = re.compile(r'"([A-Za-z0-9_./-]+\.(?:safetensors|pth|pt|ckpt|bin|gguf))"')


def _referenced() -> set[str]:
    names = set()
    for p in list((ROOT / "workflows").glob("*.json")) + list(ROOT.rglob("*.py")):
        if "comfy_ext" in p.parts or p.name == "models.py":  # the manifest names the repos' own file names
            continue
        names |= set(PAT.findall(p.read_text()))
    return names


def test_every_weight_the_code_names_is_in_the_manifest():
    known = {Path(w.path).name for w in models.WEIGHTS} | {"/".join(Path(w.path).parts[1:]) for w in models.WEIGHTS}
    # vace is a kept experiment, not a route (workflows.PROXY_VIDEO), and its files are not offered.
    unknown = {n for n in _referenced() if n not in known and Path(n).name not in known and "vace" not in n}
    assert not unknown, unknown


def test_manifest_paths_are_unique_and_grouped():
    paths = [w.path for w in models.WEIGHTS]
    assert len(paths) == len(set(paths))
    assert {w.group for w in models.WEIGHTS} <= set(models.GROUPS)
    assert "h3" not in models.ALL_GROUPS and set(models.DEFAULT_GROUPS) <= set(models.ALL_GROUPS)
    with pytest.raises(ValueError, match="unknown model group"):
        models.weights(["nope"])


def test_store_comes_from_the_env_or_the_model_paths_file(tmp_path, monkeypatch):
    monkeypatch.setenv("GIRO_MODELS", str(tmp_path / "m"))
    assert models.store() == tmp_path / "m"
    monkeypatch.delenv("GIRO_MODELS")
    assert models.store() == Path("~/models").expanduser()  # scripts/extra_model_paths.yaml


def test_fetch_downloads_what_is_missing_and_the_licenses(tmp_path):
    root = tmp_path / "store"
    (root / "vae" / "wan21").mkdir(parents=True)
    (root / "vae" / "wan21" / "wan_2.1_vae.safetensors").write_bytes(b"have")  # already there
    calls: list[tuple[str, str]] = []

    def download(repo: str, file: str, into: Path) -> Path:
        calls.append((repo, file))
        out = into / file
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"w")
        return out

    lines: list[str] = []
    done = models.fetch(["proxy"], root, log=lines.append, download=download,
                        license_files=lambda repo: ["LICENSE"] if repo == "Comfy-Org/sam3.1" else [])
    want = models.weights(["proxy"])
    assert len(done) == len(want) - 1 and ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/vae/wan_2.1_vae.safetensors") not in calls
    assert all((root / w.path).is_file() for w in want)
    assert (root / "licenses" / "Comfy-Org--sam3.1--LICENSE").is_file()
    assert not (root / ".downloads").exists()
    assert models.missing(["proxy"], root) == []
    assert "fetched" in lines[-1]
    # The H3 LoRA lands under giro's name, not the repo's.
    assert next(w for w in models.WEIGHTS if "pablodawson" in w.repo).path.endswith("minimax_h3_flf2v_orbit360_pablodawson_v1.safetensors")


def test_describe_marks_present_and_missing(tmp_path):
    root = tmp_path / "store"
    (root / "checkpoints").mkdir(parents=True)
    (root / "checkpoints" / "sam3.1_multiplex_fp16.safetensors").write_bytes(b"x" * 1024)
    text = models.describe(root)
    assert "missing  diffusion/wan22/wan2.2_fun_control_high_noise_14B_fp8_scaled.safetensors" in text
    assert re.search(r"0\.00 GB  checkpoints/sam3\.1_multiplex_fp16\.safetensors  \(SAM License\)", text)
    assert "[h3]" in text and "never part of --all" in text


def test_comfy_gets_the_store_from_the_env(tmp_path, monkeypatch):
    monkeypatch.delenv("GIRO_MODELS", raising=False)
    assert server.model_paths() == server.MODEL_PATHS
    monkeypatch.setenv("GIRO_MODELS", "/models")
    monkeypatch.setattr(server, "RUN_DIR", tmp_path / "run")
    out = server.model_paths()
    text = out.read_text()
    assert out == tmp_path / "run" / "extra_model_paths.yaml"
    assert re.search(r"^\s*base_path: /models$", text, re.MULTILINE)
    assert f"base_path: {server.ROOT / 'src' / 'giro'}" in text  # the custom nodes' path, now absolute
