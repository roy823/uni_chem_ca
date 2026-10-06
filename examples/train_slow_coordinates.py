"""Synthetic time-lagged objective demonstration, NOT a molecular trajectory.

This validates a differentiable slow-coordinate training path. It does not
train the supplied CoarseFieldEncoder for real chemistry or calibrate a bath.
"""

import json
from pathlib import Path

import torch

from unichem.models.slow import vamp2_score


def main():
    torch.set_num_threads(1)
    torch.manual_seed(53)
    # Independent paired samples from a stationary two-mode Gaussian process.
    # Slow x has correlation .98; nuisance y is independently refreshed.
    past = torch.randn(4096, 2, dtype=torch.float64)
    future = torch.randn_like(past)
    future[:, 0] = 0.98 * past[:, 0] + (1 - 0.98**2) ** 0.5 * future[:, 0]
    model = torch.nn.Linear(2, 1, bias=False).double()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.3, 0.9]], dtype=torch.float64))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.04)
    history = []
    for epoch in range(51):
        train = vamp2_score(model(past[:3072]), model(future[:3072]))
        with torch.no_grad():
            heldout = vamp2_score(model(past[3072:]), model(future[3072:]))
        history.append(
            {"epoch": epoch, "train_score": float(train.detach()), "heldout_score": float(heldout)}
        )
        if epoch < 50:
            optimizer.zero_grad()
            (-train).backward()
            optimizer.step()
    report = {
        "scope": "SYNTHETIC CORRELATED GAUSSIANS; NOT CHEMICAL KINETICS",
        "learned_weights": model.weight.detach().tolist(),
        "history": history,
    }
    path = Path("outputs/slow_training")
    path.mkdir(parents=True, exist_ok=True)
    (path / "report.json").write_text(json.dumps(report, indent=2))
    torch.save(model.state_dict(), path / "model.pt")
    print(json.dumps({**report, "history": [history[0], history[-1]]}, indent=2))


if __name__ == "__main__":
    main()
