"""Callback to print a summary of all models in the trainer's LightningModule at the start of training."""

import logging

import lightning.pytorch as pl
from tgdp.networks.ema_wrapper import EMANetworkWrapper

from ...models.diffusion.base_diffusion import BaseDiffusionModel

logger = logging.getLogger(__name__)


class ModelSummaryCallback(pl.Callback):
    """Callback to print a configurable summary of all models in the trainer's LightningModule at the start of training.

    It prints a table listing each model's & submodules' name, type, number of trainable and non-trainable parameters,
    and training mode. To facilitate readability, it indents submodules according to their hierarchy level. It treats
    diffusion models specially by showing their networks and guides without going deeper into those submodules to
    simultaneously show all relevant parts of the model and reduce clutter.
    """

    INDENT_SPACES = 2

    def __init__(
        self,
        depth: int = 2,
        report_zero_param: bool = True,
    ):
        """Initialize the callback.

        Args:
        depth (int): How deep to recurse into submodules to display.
        report_zero_param (bool): Whether to report modules with zero parameters.

        """
        self.depth = depth
        self.report_zero_param = report_zero_param

    def _format_params(self, num_params):
        """Format parameter count into human-readable string (e.g. 1.2 M, 453 K)."""
        if num_params >= 1e6:
            return f"{num_params / 1e6:.1f} M"
        elif num_params >= 1e3:
            return f"{num_params / 1e3:.0f} K"
        else:
            return str(num_params)

    def _get_model_and_note(self, module):
        """Unwrap EMA wrappers to get the true model and a note if unwrapping occurred."""
        note = ""
        if module.__class__.__name__ == "EMANetworkWrapper":
            true_module = module._original
            note = "(plus EMA copy)"
        else:
            true_module = module
        return true_module, note

    def _get_mode_str(self, module):
        """Return string 'train' if module.training is True else 'eval'."""
        return "train" if getattr(module, "training", False) else "eval"

    def _collect_rows(self, trainable_models):
        """Collect summary rows for all models and their submodules.

        Each row is a tuple: (indent_level, name, type, trainable_params, nontrainable_params, mode, note). We here
        traverse self.depth levels deep into submodules, starting at the network of any ff model, diffusion model,
        or guide model (which is inside a diffusion model).
        """
        rows = []

        def add_module(name, module, depth, indent_level=0):
            # Check for EMA model.
            if isinstance(module, EMANetworkWrapper):
                module = module._original
                note = "(plus EMA copy)"
            else:
                note = ""
            # Count parameters.
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            nontrainable = sum(p.numel() for p in module.parameters() if not p.requires_grad)
            mode = self._get_mode_str(module)

            # Skip modules with zero parameters if configured to do so.
            if self.report_zero_param or (trainable + nontrainable > 0):
                rows.append((indent_level, name, module.__class__.__name__, trainable, nontrainable, mode, note))

            # Iterate through submodules if depth allows.
            if depth > 0 and hasattr(module, "named_children"):
                for child_name, child_module in module.named_children():
                    # Special case: Diffusion models should show their network and guides without going deeper
                    # so as to both show structure of the guides without adding too much clutter.
                    if isinstance(module, BaseDiffusionModel) and child_name == "guides":
                        add_module(child_name, child_module, depth + 1, indent_level + 1)  # Add at the same depth.
                    else:
                        add_module(child_name, child_module, depth - 1, indent_level + 1)

        for name, model in trainable_models.items():
            add_module(name, model, self.depth, 0)

        return rows

    def _compute_col_widths(self, rows):
        """Compute adaptive widths for columns to align vertical separators."""
        name_width = max(
            len("Name"), max((len(" " * (self.INDENT_SPACES * indent) + name) for indent, name, *_ in rows), default=0)
        )
        type_width = max(len("Type"), max((len(r[2]) for r in rows), default=0))
        train_width = max(len("Trainable Params"), max((len(self._format_params(r[3])) for r in rows), default=0))
        nontrain_width = max(
            len("Non-Trainable Params"), max((len(self._format_params(r[4])) for r in rows), default=0)
        )
        mode_width = max(len("Mode"), max((len(r[5]) for r in rows), default=0))
        return name_width, type_width, train_width, nontrain_width, mode_width

    def _print_title(self, widths):
        """Print a horizontal separator line using the given character."""
        total_width = sum(widths) + 3 * (len(widths) - 1)  # spaces for ' | '
        half_width = int((total_width - 15) / 2)
        print("=" * half_width + " MODEL SUMMARY " + "=" * half_width)

    def _print_separator(self, widths, char="-"):
        """Print a horizontal separator line using the given character."""
        total_width = sum(widths) + 3 * (len(widths) - 1)  # spaces for ' | '
        print(char * total_width)

    def _print_line(self, indent_level, name, typ, trainable, nontrainable, mode, note, widths):
        """Print one summary row with proper padding and alignment."""
        indent_space = " " * (self.INDENT_SPACES * indent_level)
        name_col = indent_space + name
        name_str = name_col.ljust(widths[0])
        type_str = typ.ljust(widths[1])
        train_str = self._format_params(trainable).rjust(widths[2])
        nontrain_str = self._format_params(nontrainable).rjust(widths[3])
        mode_str = mode.ljust(widths[4])
        note_str = f" {note}" if note else ""
        print(f"{name_str} | {type_str} | {train_str} | {nontrain_str} | {mode_str}{note_str}")

    def on_train_start(self, trainer, pl_module):
        """Print the model summary for all trainable_models in pl_module.

        Args:
            trainer: The Trainer instance.
            pl_module: The LightningModule being trained.

        """
        trainable_models = getattr(pl_module, "trainable_models", {})

        # Collect rows of summary data
        rows = self._collect_rows(trainable_models)
        name_w, type_w, train_w, nontrain_w, mode_w = self._compute_col_widths(rows)

        # Print header row
        header = ("Name", "Type", "Trainable Params", "Non-Trainable Params", "Mode")
        widths = (name_w, type_w, train_w, nontrain_w, mode_w)
        self._print_title(widths)
        print(" | ".join(h.ljust(w) for h, w in zip(header, widths)))
        self._print_separator(widths, char="-")

        #  Totals for summary
        total_trainable_params = sum(
            [sum(p.numel() for p in m.parameters() if p.requires_grad) for m in trainable_models.values()]
        )
        total_nontrainable_params = sum(
            [sum(p.numel() for p in m.parameters() if not p.requires_grad) for m in trainable_models.values()]
        )
        # Print each collected row and accumulate totals
        for indent_level, name, typ, trainable, nontrainable, mode, note in rows:
            self._print_line(indent_level, name, typ, trainable, nontrainable, mode, note, widths)

        self._print_separator(widths, char="-")

        total_params = total_trainable_params + total_nontrainable_params
        size_mb = total_params * 4 / (1024**2)  # 4 bytes per float32 param

        # Print totals aligned with the last columns
        summary_lines = [
            f"{self._format_params(total_trainable_params)} Trainable params",
            f"{self._format_params(total_nontrainable_params)} Non-trainable params",
            f"{self._format_params(total_params)} Total params",
            f"{size_mb:.3f} Total estimated model params size (MB)",
        ]

        for line in summary_lines:
            print(line)

        # Final separator using '='
        self._print_separator(widths, char="=")
