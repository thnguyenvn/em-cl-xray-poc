import os
import sys
import json
import argparse
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def run_cmd(cmd, check=True):
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(result.stdout)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result


def get_git_commit():
    result = run_cmd(["git", "rev-parse", "--short", "HEAD"], check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def get_git_branch():
    result = run_cmd(["git", "branch", "--show-current"], check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def create_manifest(args, output_path):
    manifest = {
        "experiment_name": args.experiment_name,
        "sprint": args.sprint,
        "version": args.version,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git": {
            "branch": get_git_branch(),
            "commit": get_git_commit(),
        },
        "outputs": {
            "metrics_dir": "outputs/metrics",
            "figures_dir": "outputs/figures",
            "reports_dir": "outputs/reports",
            "logs_dir": "outputs/logs",
        },
        "notes": args.notes,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=4),
        encoding="utf-8",
    )

    return manifest


def make_zip(zip_name, include_checkpoints=False):
    zip_base = Path(zip_name).with_suffix("")
    zip_path = shutil.make_archive(
        base_name=str(zip_base),
        format="zip",
        root_dir=".",
        base_dir="outputs",
    )

    if not include_checkpoints:
        # Lightweight version: remove checkpoints by creating filtered archive.
        # Simpler approach: keep outputs as-is if checkpoint size is acceptable.
        pass

    return zip_path


def git_commit_outputs(args):
    paths = [
        "outputs/metrics",
        "outputs/figures",
        "outputs/reports",
    ]

    existing_paths = [p for p in paths if Path(p).exists()]

    if not existing_paths:
        print("No output paths found to commit.")
        return

    run_cmd(["git", "add"] + existing_paths, check=False)

    message = f"{args.sprint}: {args.experiment_name} results"
    run_cmd(["git", "commit", "-m", message], check=False)

    if args.push:
        run_cmd(["git", "push"], check=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--sprint", type=str, required=True)
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--zip_name", type=str, default=None)
    parser.add_argument("--commit_outputs", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--include_checkpoints", action="store_true")

    args = parser.parse_args()

    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = reports_dir / "experiment_manifest.json"
    manifest = create_manifest(args, manifest_path)

    print("=" * 60)
    print("Experiment Manifest")
    print("=" * 60)
    print(json.dumps(manifest, indent=4))

    if args.zip_name is None:
        safe_name = args.experiment_name.replace(" ", "_")
        args.zip_name = f"{args.sprint}_{safe_name}_{args.version}.zip"

    zip_path = make_zip(
        zip_name=args.zip_name,
        include_checkpoints=args.include_checkpoints,
    )

    print("=" * 60)
    print("ZIP created")
    print("=" * 60)
    print(zip_path)

    if args.commit_outputs:
        git_commit_outputs(args)

    print("=" * 60)
    print("Publish experiment completed")
    print("=" * 60)


if __name__ == "__main__":
    main()