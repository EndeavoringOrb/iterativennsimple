from Sequential2DSparse import Sequential2DSparse
from Sequential2D import Sequential2D
from utils.timer import Timer
import torch.nn as nn
import torch
import platform


def benchmarkModel(
    model: nn.Sequential,
    input_shape,
    n_iters: int = 50,
    n_batch: int = 32,
    additional_blocks: list[tuple] = [],
    visualize=False,
):
    x = torch.randn(
        n_iters, n_batch, *(input_shape[1:] if len(input_shape) == 4 else input_shape)
    )

    with Timer("Base"):
        for i in range(n_iters):
            model(x[i])

    torch.manual_seed(42)
    seqModel = Sequential2D(model, input_shape)
    for block in additional_blocks:
        seqModel.add_block(block)
    print(seqModel.param_info())
    if visualize:
        seqModel.visualize()

    with Timer("Seq2D"):
        for i in range(n_iters):
            seqModel(x[i])

    torch.manual_seed(42)
    seqModel = Sequential2DSparse(model, input_shape)
    for block in additional_blocks:
        seqModel.add_block(block)
    print(seqModel.param_info())

    with Timer("Seq2DSparse"):
        for i in range(n_iters):
            seqModel(x[i])

    if platform.system() == "Linux":
        seqModel = torch.compile(seqModel)

    with Timer("Seq2DSparse Compiled"):
        for i in range(n_iters):
            seqModel(x[i])


if __name__ == "__main__":
    n_iters = 1024
    n_batch = 1024

    # Test with linear layers
    print("Testing Wide Linear Model:")
    model = nn.Sequential(nn.Linear(2, 1024), nn.ReLU(), nn.Linear(1024, 1))
    benchmarkModel(model, (2,), n_iters, n_batch)

    # Test adding a block
    print("\nAdding connection block...")
    benchmarkModel(
        model,
        (2,),
        n_iters,
        n_batch,
        additional_blocks=[(2, 60, 40, 40), (0, 1026, 2, 1)],
    )

    # Test with linear layers
    print("\nTesting Deep Linear Model:")
    model = nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )
    benchmarkModel(model, (2,), n_iters, n_batch)

    # Test with conv model
    print("\nTesting Conv2D Model:")
    model = nn.Sequential(
        nn.Conv2d(1, 3, 3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3 * 4 * 4, 1),
    )
    benchmarkModel(model, (1, 1, 4, 4), n_iters, n_batch)
