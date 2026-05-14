"""
down_dataset.py
===============
One-shot script to authenticate with the Hugging Face Hub and download the
DrivingGen benchmark dataset from the hosted repository.

The dataset is stored as a HuggingFace *dataset* repository (repo_type="dataset")
under the alias "yangzhou99/DrivingGen".  Running this script clones the entire
snapshot into a local ./data directory without using symbolic links, which makes
the layout portable across cluster file-systems and shared network drives where
symlinks may be broken or disallowed.

Usage:
    python down_dataset.py

Before running, replace "your_token" with a valid HuggingFace User Access Token
that has at least read permission on the target repository.
"""

from huggingface_hub import login, snapshot_download
import os

# Authenticate with the Hugging Face Hub using a personal access token.
# add_to_git_credential=True caches the token in the system git credential
# store so that subsequent git-lfs operations (used internally by
# snapshot_download for large binary files) are also authenticated without
# prompting the user again.
login(token="your_token", add_to_git_credential=True)

snapshot_download(
    # The HuggingFace repository identifier in the format "owner/repo-name".
    repo_id="yangzhou99/DrivingGen",

    # repo_type="dataset" is required because this repo is registered as a
    # HuggingFace Dataset, not a model repo.  Omitting this argument would
    # cause the Hub client to look in the wrong namespace and fail with a
    # 404 / repository-not-found error.
    repo_type="dataset",          # Key: use repo_type for dataset repositories

    # Destination directory on the local file-system.  All repository files
    # will be written under this path, preserving the remote directory structure.
    local_dir="./data",

    # Disable symbolic links and instead copy actual file contents.
    # Symlinks can silently break on cluster/shared network storage (NFS, GPFS)
    # where inodes are not shared across mount points, leading to missing-file
    # errors at training time.  Using copies is slower but always portable.
    local_dir_use_symlinks=False, # More stable on clusters and shared drives
)
