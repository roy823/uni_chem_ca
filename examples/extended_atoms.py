"""Run after `pip install -e .` from the repository root."""

import json

import torch

from unichem.cli import run_demo

if __name__ == "__main__":
    torch.set_num_threads(1)
    print(json.dumps(run_demo(), indent=2))
