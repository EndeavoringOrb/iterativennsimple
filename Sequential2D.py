import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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


class Sequential2D:

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

        # Build giant matrix W for added connections only
        self.total_size = sum(self.sizes)
        self.W = torch.zeros(self.total_size, self.total_size)
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

    def forward(self, x):
        B = x.shape[0]

        # Process layer by layer
        current_x = x
        activations = []

        # Store input as first activation
        activations.append(current_x.reshape(B, -1))
        offset = 0
        for i, rec in enumerate(self.records):
            mod = rec["module"]

            if len(rec["weights"]) == 0:
                # Activation layer
                current_x = mod(current_x)
                # Store flattened activation
                if len(self.records[i - 1]["weights"]) == 0:
                    activations[-1] = current_x.reshape(B, -1)
                else:
                    activations.append(current_x.reshape(B, -1))
            else:
                # Parameterized layer
                offset += 1
                idx0 = self.offsets[offset + 0]
                idx1 = self.offsets[offset + 1]

                # Calculate all additional blocks
                full_activation = torch.cat(activations, dim=1)  # (B, total_size)
                additional = full_activation @ self.W.T[: full_activation.shape[1]]

                if additional.shape[1] < idx1:
                    padding = torch.zeros(B, idx1 - additional.shape[1])
                    additional = torch.cat([additional, padding], dim=1)

                current_x = mod(current_x)
                current_x += additional[:, idx0:idx1].reshape(current_x.shape)

        return current_x

    def add_block(self, block_rect):
        """Add a connection block to the weight matrix"""
        x, y, w, h = block_rect
        self.added_block_rects.append(block_rect)

        # Initialize the block with random weights
        self.W[y : y + h, x : x + w] = torch.normal(0, 2.00, (h, w))

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
        ax.set_title("Sequential2D Weight Matrix Layout")
        ax.set_xlabel("Input Features")
        ax.set_ylabel("Output Features")

        # Add grid and layer separators
        for offset in self.offsets:
            ax.axvline(x=offset, color="gray", linestyle="--", alpha=0.3)
            ax.axhline(y=-offset, color="gray", linestyle="--", alpha=0.3)

        plt.grid(True, linestyle=":", alpha=0.3)
        plt.tight_layout()
        plt.show()

    def param_info(self):
        total_weights = self.total_size * self.total_size

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

        added_trainable = torch.sum(self.W != 0).item()
        added_non_trainable = 0

        return {
            "total_matrix_size": total_weights,
            "base_trainable_params": base_trainable,
            "base_non_trainable_params": base_non_trainable,
            "added_trainable_params": added_trainable,
            "added_non_trainable_params": added_non_trainable,
        }

    def __call__(self, x):
        return self.forward(x)
