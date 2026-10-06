import pytest
import torch

from unichem import Grid, make_world


@pytest.fixture(scope="session", autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def h2():
    return make_world(
        ["H", "H"],
        [[4.63, 5.27, 5.11], [6.23, 5.27, 5.11]],
        grid=Grid((18, 18, 18), (0.6, 0.6, 0.6)),
    )


@pytest.fixture
def water():
    return make_world(
        ["O", "H", "H"],
        [[8.03, 8.07, 8.11], [9.63, 8.07, 8.11], [7.63, 9.57, 8.11]],
        grid=Grid((28, 28, 28), (0.6, 0.6, 0.6)),
    )
