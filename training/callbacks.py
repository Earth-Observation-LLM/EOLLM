"""
callbacks.py — Trainer callbacks for loss logging, W&B metrics, and plotting.
"""

from __future__ import annotations

from pathlib import Path

from transformers import TrainerCallback


# ---------------------------------------------------------------------------
# Loss logger — writes CSV incrementally
# ---------------------------------------------------------------------------


class LossLogger(TrainerCallback):
    """Logs train and eval loss to CSV files for post-hoc plotting."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.steps: list[int] = []
        self.losses: list[float] = []
        self.lrs: list[float] = []
        self.eval_steps: list[int] = []
        self.eval_losses: list[float] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.steps.append(state.global_step)
            self.losses.append(logs["loss"])
            self.lrs.append(logs.get("learning_rate", 0))
            with open(self.run_dir / "training_loss.csv", "w") as f:
                f.write("step,loss,learning_rate\n")
                for s, l, r in zip(self.steps, self.losses, self.lrs):
                    f.write(f"{s},{l},{r}\n")

        if logs and "eval_loss" in logs:
            self.eval_steps.append(state.global_step)
            self.eval_losses.append(logs["eval_loss"])
            with open(self.run_dir / "eval_loss.csv", "w") as f:
                f.write("step,eval_loss\n")
                for s, l in zip(self.eval_steps, self.eval_losses):
                    f.write(f"{s},{l}\n")


# ---------------------------------------------------------------------------
# W&B metrics callback
# ---------------------------------------------------------------------------


class WandbCallback(TrainerCallback):
    """Logs extra metrics to W&B beyond what SFTTrainer logs natively.

    - Per-topic accuracy at eval time (on a small val subset)
    - Uploads training curves plot at the end
    """

    def __init__(
        self,
        val_records: list[dict],
        base_dir: str,
        max_edge: int,
        run_dir: Path,
        eval_n: int = 200,
    ):
        self.val_records = val_records
        self.base_dir = base_dir
        self.max_edge = max_edge
        self.run_dir = run_dir
        self.eval_n = eval_n
        self._eval_count = 0

    def on_evaluate(self, args, state, control, model=None, processing_class=None, **kwargs):
        """Run per-topic accuracy after each trainer eval step."""
        try:
            import wandb
            if wandb.run is None:
                return

            from evaluation import compute_topic_accuracy
            from unsloth import FastVisionModel

            self._eval_count += 1
            accuracy = compute_topic_accuracy(
                model, processing_class, self.val_records,
                self.base_dir, self.max_edge,
                n=self.eval_n, seed=state.global_step,
            )

            # Log overall accuracy
            wandb.log({"eval/accuracy": accuracy["overall"]}, step=state.global_step)

            # Log per-topic
            for topic, info in accuracy["per_topic"].items():
                wandb.log({f"eval/acc_{topic}": info["acc"]}, step=state.global_step)

            # Back to training mode
            FastVisionModel.for_training(model)

        except Exception as e:
            print(f"[wandb callback] Error during eval accuracy: {e}")

    def on_train_end(self, args, state, control, **kwargs):
        """Upload training curves plot as artifact."""
        try:
            import wandb
            if wandb.run is None:
                return
            plot_path = self.run_dir / "training_curves.png"
            if plot_path.exists():
                wandb.log({"training_curves": wandb.Image(str(plot_path))})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_training_curves(loss_logger: LossLogger, run_dir: Path, profile_name: str):
    """Generate training curves plot from logged data."""
    if not loss_logger.steps:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    has_eval = len(loss_logger.eval_steps) > 0
    n_plots = 3 if has_eval else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

    # Train + eval loss
    axes[0].plot(loss_logger.steps, loss_logger.losses, "b-", linewidth=1.2, alpha=0.7, label="Train")
    if has_eval:
        axes[0].plot(loss_logger.eval_steps, loss_logger.eval_losses, "ro-",
                     markersize=6, linewidth=2, label="Validation")
    axes[0].axhline(y=1.386, color="gray", linestyle="--", alpha=0.4, label="Random (ln4)")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # LR schedule
    axes[1].plot(loss_logger.steps, loss_logger.lrs, "r-", linewidth=1.5)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("Learning Rate Schedule")
    axes[1].grid(True, alpha=0.3)

    # Loss distribution (second half of training)
    if has_eval and len(axes) > 2:
        mid = len(loss_logger.losses) // 2
        if mid > 0:
            axes[2].hist(loss_logger.losses[mid:], bins=30, color="steelblue", alpha=0.8, edgecolor="white")
            mean_loss = sum(loss_logger.losses[mid:]) / len(loss_logger.losses[mid:])
            axes[2].axvline(x=mean_loss, color="red", linestyle="--", label=f"Mean: {mean_loss:.3f}")
            axes[2].set_xlabel("Loss")
            axes[2].set_ylabel("Count")
            axes[2].set_title(f"Loss Distribution (steps {loss_logger.steps[mid]}–{loss_logger.steps[-1]})")
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"EOLLM Vision SFT — Qwen3.5-4B LoRA ({profile_name})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_path = run_dir / "training_curves.png"
    fig.savefig(str(plot_path), dpi=150)
    plt.close()
    print(f"Training curves saved to {plot_path}")
