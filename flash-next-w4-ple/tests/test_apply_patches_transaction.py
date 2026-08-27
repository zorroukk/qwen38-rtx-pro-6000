#!/usr/bin/env python3
"""Exercise installer integrity, rollback, linked-worktree, and LF gates."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path


EXPECTED_COMMIT = "73a255206f916366c8d26d4022f82ddfb0ab558d"
QWEN_REL = Path("python/sglang/srt/models/qwen4_exp.py")
QSA_REL = Path("python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py")
LOADER_REL = Path("python/sglang/srt/models/qwen4_ple_w4_sidecar.py")
BASE_HASHES = {
    QWEN_REL: "f406977eb2373937393241f453477867f7dc943bd4839216db8fe66fa9f921d8",
    QSA_REL: "c959835d05d0f395ad7eae4330cf264af9f6f7c1bff3d45a39bb953d2536f5f2",
}
FINAL_HASHES = {
    QWEN_REL: "b54de0a07a16a7a3070aabead3c53b80f108d89528c30763ca8e032f330d97ac",
    QSA_REL: "584e2acdce11c6e1e6dc50b9f61ca8018a3634e1bafb3d079b367fc30d1f7634",
    LOADER_REL: "9dbace396f69ca2a319c7ae9cf74380549b62b6cdcc9f88135776552c245bb68",
}


def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, **kwargs)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_hashes(root: Path, expected: dict[Path, str]) -> None:
    for relative, digest in expected.items():
        actual = sha256_file(root / relative)
        require(actual == digest, f"{relative}: {actual} != {digest}")


def assert_pristine(worktree: Path) -> None:
    status = run(
        ["git", "-C", str(worktree), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
    ).stdout
    require(status == "", f"target is dirty after failed install:\n{status}")
    assert_hashes(worktree, BASE_HASHES)
    require(not (worktree / LOADER_REL).exists(), "sidecar loader remains")


def verify_checksum_manifest(release_dir: Path) -> None:
    for line in (release_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        actual = sha256_file(release_dir / relative)
        require(actual == expected, f"{relative}: {actual} != {expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sglang-repo", type=Path, required=True)
    args = parser.parse_args()

    release_dir = Path(__file__).resolve().parents[1]
    repository = release_dir.parent
    sglang_repo = args.sglang_repo.resolve(strict=True)

    with tempfile.TemporaryDirectory(prefix="qwen4-ple-installer-test-") as name:
        test_root = Path(name)
        autocrlf_clone = test_root / "release-autocrlf"
        worktree = test_root / "sglang-linked-worktree"
        run(
            [
                "git",
                "-c",
                "core.autocrlf=true",
                "clone",
                "--quiet",
                "--no-local",
                str(repository),
                str(autocrlf_clone),
            ]
        )
        cloned_release = autocrlf_clone / release_dir.name
        verify_checksum_manifest(cloned_release)

        run(
            [
                "git",
                "-C",
                str(sglang_repo),
                "worktree",
                "add",
                "--quiet",
                "--detach",
                str(worktree),
                EXPECTED_COMMIT,
            ]
        )
        try:
            helper = cloned_release / "scripts/apply_patches_transaction.py"

            tampered_patch = cloned_release / "patches/0001-qwen4-exp-w4-ple.patch"
            with tampered_patch.open("ab") as writer:
                writer.write(b"\n# tamper\n")
            refused = subprocess.run(
                [sys.executable, str(helper), str(worktree)], text=True
            )
            require(refused.returncode != 0, "tampered release input was accepted")
            assert_pristine(worktree)
            run(
                [
                    "git",
                    "-C",
                    str(autocrlf_clone),
                    "checkout",
                    "--",
                    "flash-next-w4-ple/patches/0001-qwen4-exp-w4-ple.patch",
                ]
            )

            if os.name != "nt":
                fake_bin = test_root / "fake-bin"
                fake_bin.mkdir()
                fake_python = fake_bin / "python3"
                fake_python.write_text("#!/bin/sh\nexit 97\n", encoding="utf-8")
                fake_python.chmod(0o755)
                environment = os.environ.copy()
                environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
                broken = subprocess.run(
                    [str(cloned_release / "scripts/apply_patches.sh"), str(worktree)],
                    env=environment,
                    text=True,
                )
                require(
                    broken.returncode == 69,
                    f"broken Python returned {broken.returncode}, expected 69",
                )
                assert_pristine(worktree)

            for stage in ("staged", "qwen", "qsa", "loader"):
                environment = os.environ.copy()
                environment["SGLANG_QWEN4_PLE_PATCH_TEST_FAIL_AFTER"] = stage
                failed = subprocess.run(
                    [sys.executable, str(helper), str(worktree)],
                    env=environment,
                    text=True,
                )
                require(
                    failed.returncode != 0,
                    f"injection {stage} unexpectedly passed",
                )
                assert_pristine(worktree)

            success = run(
                [sys.executable, str(helper), str(worktree)], capture_output=True
            )
            require(
                f"PATCH_SET_VERIFIED {EXPECTED_COMMIT}" in success.stdout,
                "success marker missing",
            )
            assert_hashes(worktree, FINAL_HASHES)
            print("PATCH_TRANSACTION_TESTS_OK")
        finally:
            run(
                [
                    "git",
                    "-C",
                    str(sglang_repo),
                    "worktree",
                    "remove",
                    "--force",
                    str(worktree),
                ]
            )


if __name__ == "__main__":
    main()
