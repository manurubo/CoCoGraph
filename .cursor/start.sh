#!/usr/bin/env bash
# Per-boot reconciliation for CoCoGraph.
# The environment build caches the Git LFS objects in .git/lfs, but Cloud Agent
# boots re-check-out the working tree with GIT_LFS_SKIP_SMUDGE=1, which leaves
# LFS-tracked files as pointer stubs. Materialize the dataset needed by the
# sampling/evaluation demos on every boot. This is fast and offline when the
# object is already cached (the build case); it falls back to a network pull for
# just-in-time agents that skipped the build.
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="Data/molecules_lt70atoms_annotated.csv"

if command -v git-lfs >/dev/null 2>&1; then
  # Try a local (offline) checkout from the LFS object cache first.
  git lfs checkout "$DATASET" 2>/dev/null || true

  # If the working-tree file is still an LFS pointer, fetch it from the remote.
  if head -c 64 "$DATASET" 2>/dev/null | grep -q "git-lfs.github.com"; then
    git lfs pull --include="$DATASET" || true
  fi
fi
