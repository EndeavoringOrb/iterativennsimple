import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.profiler import profile


class BlockSparseMatrix:
    """Block-sparse matrix representation storing blocks as (x,y,width,height) and values"""

    def __init__(self, total_rows, total_cols):
        self.total_rows = total_rows
        self.total_cols = total_cols
        self.blocks = []  # List of (x, y, width, height, values) tuples

    @profile
    def add_block(self, x, y, width, height, values=None):
        """Add a block at position (x,y) with given dimensions and values"""
        if values is None:
            # Initialize with random values
            values = torch.normal(0, 2.00, (height, width))
        else:
            values = torch.as_tensor(values)
            assert values.shape == (
                height,
                width,
            ), f"Values shape {values.shape} doesn't match block dimensions ({height}, {width})"

        self.blocks.append((x, y, width, height, values))

    @profile
    def matmul(self, x):
        """Multiply block-sparse matrix with input vector x"""
        # x shape: (batch_size, total_cols)
        batch_size = x.shape[0]
        x_cols = x.shape[1]
        result = torch.zeros(
            batch_size, self.total_rows, device=x.device, dtype=x.dtype
        )

        for block_x, block_y, width, height, values in self.blocks:
            # Extract the relevant portion of input
            x_block = x[:, block_x : block_x + width]  # (batch_size, width)

            # Multiply with block values
            output_block = x_block @ values.T  # (batch_size, height)

            # Add to result
            result[:, block_y : block_y + height] += output_block

        return result

    @profile
    def matmul_partial(self, x, x_cols, idx0, idx1, result: torch.Tensor):
        """Multiply block-sparse matrix with input vector x, returning rows idx0:idx1"""
        for block_x, block_y, width, height, values in self.blocks:
            # Skip blocks that don't contribute to the output range
            if block_y >= idx1 or block_y + height <= idx0:
                continue

            # Skip blocks that are entirely beyond the input
            if block_x >= x_cols:
                continue

            # Calculate the intersection of the block with the input and output ranges
            input_start = block_x
            input_end = min(block_x + width, x_cols)
            input_width = input_end - input_start

            if input_width <= 0:
                continue

            output_start = max(block_y, idx0)
            output_end = min(block_y + height, idx1)

            if output_start >= output_end:
                continue

            # Extract relevant portions
            x_block = x[:, input_start:input_end]  # (batch_size, input_width)

            # Get the relevant part of the values matrix
            values_row_start = output_start - block_y
            values_row_end = output_end - block_y
            values_col_end = input_width

            values_subset = values[values_row_start:values_row_end, :values_col_end]

            # Multiply and accumulate
            output_block = x_block @ values_subset.T

            # Add to result at the correct position
            result[:, output_start:output_end] += output_block

        return result

    def get_nonzero_count(self):
        """Get total number of non-zero elements"""
        return sum(width * height for _, _, width, height, _ in self.blocks)

    def get_blocks_info(self):
        """Get information about all blocks"""
        return [(x, y, width, height) for x, y, width, height, _ in self.blocks]

    def to_dense(self):
        """Convert to dense matrix (for debugging/visualization)"""
        dense = torch.zeros(self.total_rows, self.total_cols)
        for x, y, width, height, values in self.blocks:
            dense[y : y + height, x : x + width] = values
        return dense


class ShapeTracker:
    def __init__(self, model):
        self.model: nn.Sequential = model
        self.records = []

        self.hooks = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # leaf modules only
                hook = module.register_forward_hook(self.make_hook(name, module))
                self.hooks.append(hook)

    def make_hook(self, name, module):
        def hook(mod, inp, out):
            input_shape = (
                tuple(inp[0].shape) if isinstance(inp, tuple) else tuple(inp.shape)
            )
            output_shape = tuple(out.shape)
            weights = mod.state_dict()  # can be empty if module has no params
            self.records.append(
                {
                    "name": name,
                    "module": module,
                    "input_shape": input_shape,
                    "output_shape": output_shape,
                    "weights": weights,
                    "forward_func": mod.forward,  # function pointer
                }
            )

        return hook

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_records(self):
        return self.records


def flat_size(shape):
    flat_size = 1
    for item in shape:
        if isinstance(item, tuple):
            for it2 in item:
                flat_size *= it2
        else:
            flat_size *= item
    return flat_size


class Sequential2DSparse:
    @profile
    def __init__(self, seq: nn.Sequential, input_shape):
        tracker = ShapeTracker(seq)
        x = torch.randn(*input_shape)
        seq(x)
        tracker.clear_hooks()
        self.records = tracker.get_records()

        # Build activation layout
        self.shapes = [input_shape]
        self.sizes = [flat_size(input_shape)]
        self.offsets = [0]

        current_offset = 0
        for rec in self.records:
            mod = rec["module"]

            if len(rec["weights"]) == 0:
                continue

            current_offset += self.sizes[-1]
            self.offsets.append(current_offset)
            if isinstance(mod, nn.Conv2d):
                # For Conv2d, we need to track the unfolded input representation
                out_shape = rec["output_shape"]
                self.shapes.append(out_shape)
                self.sizes.append(flat_size(out_shape))
            else:
                # Linear and other layers
                out_shape = rec["output_shape"]
                self.shapes.append(out_shape)
                self.sizes.append(flat_size(out_shape))

        # Final offset
        current_offset += self.sizes[-1]
        self.offsets.append(current_offset)

        # Build block-sparse matrix W for added connections only
        self.total_size = sum(self.sizes)
        self.W = BlockSparseMatrix(self.total_size, self.total_size)
        self.base_block_rects = []
        self.added_block_rects = []

        # Track base model blocks for visualization
        shape_idx = 0
        for rec in self.records:
            mod = rec["module"]
            if len(rec["weights"]) == 0:
                # Skip activations for block tracking
                continue

            in_off = self.offsets[shape_idx]
            out_off = self.offsets[shape_idx + 1]

            if isinstance(mod, nn.Conv2d):
                # Conv2d block representation
                in_size = self.sizes[shape_idx]
                out_size = self.sizes[shape_idx + 1]
                self.base_block_rects.append((in_off, out_off, in_size, out_size))
            elif isinstance(mod, nn.Linear):
                # Linear layer
                w = mod.weight
                self.base_block_rects.append((in_off, out_off, w.shape[1], w.shape[0]))

            shape_idx += 1

    @profile
    def forward(self, x):
        B = x.shape[0]

        # Process layer by layer
        current_x = x
        activations = torch.zeros(B, self.total_size)
        additional = torch.zeros(B, self.total_size)
        activations_idx = 0

        # Store input as first activation
        input_flat_size = current_x.numel() // B
        activations[:, :input_flat_size] = current_x.view(B, -1)
        activations_idx += input_flat_size
        offset = 0
        for i, rec in enumerate(self.records):
            mod = rec["module"]

            if len(rec["weights"]) == 0:
                # Activation layer
                current_x = mod(current_x)
                # Store flattened activation
                if isinstance(mod, nn.Flatten):
                    pass
                else:
                    current_flat_size = current_x.numel() // B
                    activations[
                        :, activations_idx : activations_idx + current_flat_size
                    ] = current_x.view(B, -1)
                    activations_idx += current_x.numel() // B
            else:
                # Parameterized layer
                offset += 1
                idx0 = self.offsets[offset + 0]
                idx1 = self.offsets[offset + 1]

                # Calculate all additional blocks using block-sparse matrix
                self.W.matmul_partial(
                    activations, activations_idx, idx0, idx1, additional
                )

                current_x = mod(current_x)
                additional_slice = additional[:, idx0:idx1]
                if current_x.shape == additional_slice.shape:
                    current_x = current_x + additional_slice
                else:
                    current_x = current_x + additional_slice.reshape(current_x.shape)

        return current_x

    @profile
    def add_block(self, block_rect, values=None):
        """Add a connection block to the weight matrix"""
        x, y, w, h = block_rect
        self.added_block_rects.append(block_rect)

        # Add block to sparse matrix
        self.W.add_block(x, y, w, h, values)

    @profile
    def visualize(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot base blocks in blue
        for i, (x, y, w, h) in enumerate(self.base_block_rects):
            rect = patches.Rectangle(
                (x, -y - h),  # flip vertically for better layout
                w,
                h,
                linewidth=1,
                edgecolor="blue",
                facecolor="lightblue",
                alpha=0.7,
            )
            ax.add_patch(rect)
            ax.text(
                x + w / 2,
                -y - h / 2,
                f"B{i}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

        # Plot added blocks in red
        for i, (x, y, w, h) in enumerate(self.added_block_rects):
            rect = patches.Rectangle(
                (x, -y - h),
                w,
                h,
                linewidth=1,
                edgecolor="red",
                facecolor="lightcoral",
                alpha=0.7,
            )
            ax.add_patch(rect)
            ax.text(
                x + w / 2,
                -y - h / 2,
                f"A{i}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )

        ax.set_xlim(0, self.total_size)
        ax.set_ylim(-self.total_size, 0)
        ax.set_aspect("equal")
        ax.set_title("Sequential2D Block-Sparse Weight Matrix Layout")
        ax.set_xlabel("Input Features")
        ax.set_ylabel("Output Features")

        # Add grid and layer separators
        for offset in self.offsets:
            ax.axvline(x=offset, color="gray", linestyle="--", alpha=0.3)
            ax.axhline(y=-offset, color="gray", linestyle="--", alpha=0.3)

        plt.grid(True, linestyle=":", alpha=0.3)
        plt.tight_layout()
        plt.show()

    @profile
    def param_info(self):
        total_matrix_size = self.total_size * self.total_size

        base_trainable = 0
        base_non_trainable = 0
        for rec in self.records:
            mod = rec["module"]
            if hasattr(mod, "weight") and mod.weight is not None:
                if mod.weight.requires_grad:
                    base_trainable += mod.weight.numel()
                else:
                    base_non_trainable += mod.weight.numel()
            if hasattr(mod, "bias") and mod.bias is not None:
                if mod.bias.requires_grad:
                    base_trainable += mod.bias.numel()
                else:
                    base_non_trainable += mod.bias.numel()

        added_trainable = self.W.get_nonzero_count()
        added_non_trainable = 0

        return {
            "total_matrix_size": total_matrix_size,
            "base_trainable_params": base_trainable,
            "base_non_trainable_params": base_non_trainable,
            "added_trainable_params": added_trainable,
            "added_non_trainable_params": added_non_trainable,
            "sparse_efficiency": f"{added_trainable}/{total_matrix_size} ({100*added_trainable/total_matrix_size:.2f}%)",
        }

    def __call__(self, x):
        return self.forward(x)
