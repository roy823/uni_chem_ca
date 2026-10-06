"""This trains a real network on SYNTHETIC labels, not a quantum-chemistry dataset."""

import json

import torch

from unichem.training import train_synthetic

if __name__ == "__main__":
    torch.set_num_threads(1)
    print(json.dumps(train_synthetic(), indent=2))
