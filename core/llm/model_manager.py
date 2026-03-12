"""Phase 44 — Model download and management for Ariaska_RL.

Provides utilities to download, discover, and recommend LLM models for both
llama-cpp (GGUF) and HuggingFace Transformers backends.  Intended as the
single source of truth for which models are available locally and which ones
to fetch when a user needs a new one.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from rich.console import Console
from rich.table import Table

log = logging.getLogger(__name__)
console = Console()

# ---------------------------------------------------------------------------
# Recommended model catalogue
# ---------------------------------------------------------------------------

RECOMMENDED_MODELS: dict[str, dict] = {
    "qwen-7b-gguf": {
        "repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "filename": "*Q4_K_M.gguf",
        "vram_gb": 5,
        "backend": "llama-cpp",
    },
    "qwen-7b-hf": {
        "repo": "Qwen/Qwen2.5-7B-Instruct",
        "vram_gb": 14,
        "backend": "transformers",
    },
    "qwen-14b-gguf": {
        "repo": "Qwen/Qwen2.5-14B-Instruct-GGUF",
        "filename": "*Q4_K_M.gguf",
        "vram_gb": 9,
        "backend": "llama-cpp",
    },
    "mistral-7b-gguf": {
        "repo": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        "filename": "*Q4_K_M.gguf",
        "vram_gb": 5,
        "backend": "llama-cpp",
    },
    "phi-3.5-gguf": {
        "repo": "bartowski/Phi-3.5-mini-instruct-GGUF",
        "filename": "*Q4_K_M.gguf",
        "vram_gb": 3,
        "backend": "llama-cpp",
    },
}

_DEFAULT_SEARCH_DIRS: list[Path] = [
    Path.home() / "models",
    Path("models"),
    Path("/models"),
]

_DEFAULT_DOWNLOAD_DIR: Path = Path.home() / "models"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_model_info(model_alias: str) -> dict:
    """Return catalogue entry for *model_alias*.

    Args:
        model_alias: Key in ``RECOMMENDED_MODELS``.

    Raises:
        KeyError: If the alias is not found.
    """
    if model_alias not in RECOMMENDED_MODELS:
        available = ", ".join(sorted(RECOMMENDED_MODELS))
        raise KeyError(
            f"Unknown model alias '{model_alias}'. Available: {available}"
        )
    return {**RECOMMENDED_MODELS[model_alias], "alias": model_alias}


def download_model(
    model_alias: str,
    target_dir: Path | None = None,
) -> Path:
    """Download a model from the HuggingFace Hub.

    Args:
        model_alias: Key in ``RECOMMENDED_MODELS``.
        target_dir: Directory to store the download.  Defaults to ``~/models``.

    Returns:
        Path to the downloaded file (GGUF) or snapshot directory (HF).
    """
    info = get_model_info(model_alias)
    dest = target_dir or _DEFAULT_DOWNLOAD_DIR
    dest.mkdir(parents=True, exist_ok=True)

    repo_id: str = info["repo"]
    backend: str = info["backend"]

    log.info("Downloading %s from %s → %s", model_alias, repo_id, dest)
    console.print(
        f"[bold cyan]Downloading[/] [green]{model_alias}[/] "
        f"from [yellow]{repo_id}[/] …"
    )

    if backend == "llama-cpp":
        filename_pattern: str = info["filename"]
        path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename_pattern,
                local_dir=str(dest / model_alias),
            )
        )
        console.print(f"[bold green]✓[/] Saved to {path}")
        return path

    # HuggingFace Transformers — full snapshot
    snapshot_dir = dest / model_alias
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(snapshot_dir),
    )
    console.print(f"[bold green]✓[/] Snapshot saved to {snapshot_dir}")
    return snapshot_dir


def list_local_models(
    search_dirs: list[Path] | None = None,
) -> list[dict]:
    """Scan directories for locally available model files.

    Looks for ``*.gguf`` and ``*.safetensors`` files.

    Args:
        search_dirs: Directories to scan.  Defaults to
            ``~/models``, ``./models``, ``/models``.

    Returns:
        List of dicts with keys ``name``, ``path``, ``size_gb``, ``format``.
    """
    dirs = search_dirs or _DEFAULT_SEARCH_DIRS
    found: list[dict] = []
    seen_paths: set[Path] = set()

    for directory in dirs:
        if not directory.is_dir():
            continue
        for pattern, fmt in (("**/*.gguf", "gguf"), ("**/*.safetensors", "safetensors")):
            for p in directory.glob(pattern):
                resolved = p.resolve()
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)
                size_gb = resolved.stat().st_size / (1024**3)
                found.append(
                    {
                        "name": p.stem,
                        "path": resolved,
                        "size_gb": round(size_gb, 2),
                        "format": fmt,
                    }
                )

    found.sort(key=lambda m: m["name"])
    return found


def detect_hardware() -> dict:
    """Detects available hardware (CPU or GPU) and VRAM capacity.
    
    Returns:
        Dict detailing if a GPU is present, available VRAM in GB, and the recommended backend.
    """
    import subprocess
    has_gpu = False
    vram_gb = 0.0
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            has_gpu = True
            try:
                # Sum VRAM across all GPUs
                vram_total_mb = sum(int(x.strip()) for x in result.stdout.strip().split('\n') if x.strip().isdigit())
                vram_gb = vram_total_mb / 1024.0
            except ValueError:
                pass
    except FileNotFoundError:
        pass
    
    return {
        "has_gpu": has_gpu,
        "vram_gb": round(vram_gb, 2),
        "device": "cuda" if has_gpu else "cpu",
        "recommended_backend": "llama-cpp" if has_gpu else "ollama"
    }


def get_recommended_model(vram_gb: float) -> str:
    """Recommend the best model alias that fits in *vram_gb*.

    Picks the largest model (by VRAM requirement) that still fits within the
    available VRAM budget.

    Args:
        vram_gb: Available GPU VRAM in gigabytes.

    Returns:
        Model alias string.

    Raises:
        RuntimeError: If no model fits the VRAM budget.
    """
    candidates = [
        (alias, info)
        for alias, info in RECOMMENDED_MODELS.items()
        if info["vram_gb"] <= vram_gb
    ]
    if not candidates:
        min_req = min(i["vram_gb"] for i in RECOMMENDED_MODELS.values())
        raise RuntimeError(
            f"No model fits in {vram_gb} GB VRAM. "
            f"Minimum requirement is {min_req} GB."
        )
    # Prefer the largest model that fits
    candidates.sort(key=lambda c: c[1]["vram_gb"], reverse=True)
    return candidates[0][0]


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _cli_download(args: argparse.Namespace) -> None:
    target = Path(args.target) if args.target else None
    download_model(args.alias, target_dir=target)


def _cli_list(args: argparse.Namespace) -> None:  # noqa: ARG001
    models = list_local_models()
    if not models:
        console.print("[yellow]No local models found.[/]")
        return

    table = Table(title="Local Models")
    table.add_column("Name", style="green")
    table.add_column("Format", style="cyan")
    table.add_column("Size (GB)", justify="right")
    table.add_column("Path", style="dim")

    for m in models:
        table.add_row(m["name"], m["format"], str(m["size_gb"]), str(m["path"]))

    console.print(table)


def _cli_recommend(args: argparse.Namespace) -> None:
    alias = get_recommended_model(args.vram)
    info = get_model_info(alias)
    console.print(
        f"[bold green]Recommended:[/] [cyan]{alias}[/]  "
        f"(~{info['vram_gb']} GB VRAM, backend={info['backend']})"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model_manager",
        description="Ariaska_RL — Model download & management (Phase 44)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="Download a model from HuggingFace Hub")
    dl.add_argument("alias", choices=sorted(RECOMMENDED_MODELS), help="Model alias")
    dl.add_argument("--target", default=None, help="Target directory (default: ~/models)")
    dl.set_defaults(func=_cli_download)

    ls = sub.add_parser("list", help="List locally available models")
    ls.set_defaults(func=_cli_list)

    rec = sub.add_parser("recommend", help="Recommend a model for your VRAM budget")
    rec.add_argument("--vram", type=float, required=True, help="Available VRAM in GB")
    rec.set_defaults(func=_cli_recommend)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parsed = _build_parser().parse_args()
    parsed.func(parsed)
