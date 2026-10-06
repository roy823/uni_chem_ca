"""Run from an installed checkout: python examples/coupled_world.py"""

import json

import torch

from unichem.demos import run_coupled_demo

if __name__ == "__main__":
    torch.set_num_threads(1)
    print(json.dumps(run_coupled_demo(), ensure_ascii=False, indent=2))
