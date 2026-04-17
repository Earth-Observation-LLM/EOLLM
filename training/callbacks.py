"""
callbacks.py — Trainer callbacks for loss logging, W&B metrics, and plotting.

Eval schedule:
  - Every half-epoch: quick eval on 100 random val samples (accuracy + per-topic)
  - Every full epoch: complete eval on entire val set (6,984 samples)
  - Both are logged as separate W&B metric groups (quick/ vs full/)
  - An epoch verdict markdown is uploaded as a W&B artifact after each full eval
"""

from __future__ import annotations

import math
import time
from pathlib import Path

from transformers import TrainerCallback


# ---------------------------------------------------------------------------
# Loss logger — writes CSV incrementally
# ---------------------------------------------------------------------------


class LossLogger(TrainerCallback):
    """Logs train and eval loss to CSV files."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.steps: list[int] = []
        self.losses: list[float] = []
        self.lrs: list[float] = []
        self.grad_norms: list[float] = []
        self.eval_steps: list[int] = []
        self.eval_losses: list[float] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.steps.append(state.global_step)
            self.losses.append(logs["loss"])
            self.lrs.append(logs.get("learning_rate", 0))
            self.grad_norms.append(logs.get("grad_norm", 0))
            with open(self.run_dir / "training_loss.csv", "w") as f:
                f.write("step,loss,learning_rate,grad_norm\n")
                for s, l, r, g in zip(self.steps, self.losses, self.lrs, self.grad_norms):
                    f.write(f"{s},{l},{r},{g}\n")

        if logs and "eval_loss" in logs:
            self.eval_steps.append(state.global_step)
            self.eval_losses.append(logs["eval_loss"])
            with open(self.run_dir / "eval_loss.csv", "w") as f:
                f.write("step,eval_loss\n")
                for s, l in zip(self.eval_steps, self.eval_losses):
                    f.write(f"{s},{l}\n")


# ---------------------------------------------------------------------------
# Smart eval callback — quick (half-epoch) + full (epoch) + W&B
# ---------------------------------------------------------------------------


class EvalCallback(TrainerCallback):
    """Runs accuracy evaluation at smart intervals and logs to W&B.

    - Half-epoch: 100 random samples → quick/accuracy, quick/acc_{topic}
    - Full epoch: entire val set → full/accuracy, full/acc_{topic}, epoch verdict
    - Baseline (base model) scores logged once at init for comparison
    - Uploads epoch_verdict.md as W&B artifact
    """

    def __init__(
        self,
        val_records: list[dict],
        base_dir: str,
        max_edge: int,
        run_dir: Path,
        steps_per_epoch: int,
        baseline_accuracy: dict | None = None,
        use_wandb: bool = False,
    ):
        self.val_records = val_records
        self.base_dir = base_dir
        self.max_edge = max_edge
        self.run_dir = run_dir
        self.steps_per_epoch = steps_per_epoch
        self.baseline = baseline_accuracy
        self.use_wandb = use_wandb

        self._half_epoch_step = max(1, steps_per_epoch // 2)
        self._last_quick_eval = -1
        self._last_full_eval = -1
        self._epoch_count = 0

        # History for trend detection
        self.quick_history: list[dict] = []  # [{step, accuracy, per_topic}]
        self.full_history: list[dict] = []

    def on_step_end(self, args, state, control, model=None, processing_class=None, **kwargs):
        step = state.global_step
        epoch_progress = step / self.steps_per_epoch
        current_epoch = int(epoch_progress)

        # Quick eval every half epoch
        half_epoch_num = int(epoch_progress * 2)
        if half_epoch_num > self._last_quick_eval:
            self._last_quick_eval = half_epoch_num
            self._run_quick_eval(model, processing_class, step, epoch_progress)

        # Full eval every epoch boundary
        if current_epoch > self._last_full_eval and step >= self.steps_per_epoch:
            self._last_full_eval = current_epoch
            self._epoch_count = current_epoch
            self._run_full_eval(model, processing_class, step, current_epoch)

    def _run_quick_eval(self, model, tokenizer, step, epoch_progress):
        """100 random val samples — fast per-topic accuracy snapshot."""
        from evaluation import compute_topic_accuracy
        from unsloth import FastVisionModel

        print(f"\n[eval] Quick eval at step {step} (epoch {epoch_progress:.2f})...")
        t0 = time.time()
        accuracy = compute_topic_accuracy(
            model, tokenizer, self.val_records,
            self.base_dir, self.max_edge,
            n=100, seed=step,
        )
        elapsed = time.time() - t0

        self.quick_history.append({"step": step, "epoch": epoch_progress, **accuracy})
        print(f"[eval] Quick: {accuracy['overall']:.1%} ({accuracy['n_correct']}/{accuracy['n_total']}) in {elapsed:.0f}s")

        if self.use_wandb:
            self._log_wandb_accuracy(accuracy, step, prefix="quick")

        FastVisionModel.for_training(model)

    def _run_full_eval(self, model, tokenizer, step, epoch):
        """Full val set — comprehensive accuracy + epoch verdict."""
        from evaluation import compute_topic_accuracy
        from unsloth import FastVisionModel

        print(f"\n[eval] FULL eval at step {step} (epoch {epoch})...")
        t0 = time.time()
        accuracy = compute_topic_accuracy(
            model, tokenizer, self.val_records,
            self.base_dir, self.max_edge,
            n=len(self.val_records), seed=step,
        )
        elapsed = time.time() - t0

        self.full_history.append({"step": step, "epoch": epoch, **accuracy})
        print(f"[eval] Full: {accuracy['overall']:.1%} ({accuracy['n_correct']}/{accuracy['n_total']}) in {elapsed:.0f}s")

        for topic, info in sorted(accuracy["per_topic"].items()):
            print(f"  {topic}: {info['acc']:.0%} ({info['correct']}/{info['n']})")

        # Write epoch verdict
        verdict_path = self.run_dir / f"epoch_{epoch}_verdict.md"
        self._write_verdict(accuracy, epoch, step, elapsed, verdict_path)

        if self.use_wandb:
            self._log_wandb_accuracy(accuracy, step, prefix="full")
            self._upload_verdict(verdict_path, epoch)

        FastVisionModel.for_training(model)

    def _log_wandb_accuracy(self, accuracy: dict, step: int, prefix: str):
        """Log accuracy metrics to W&B."""
        try:
            import wandb
            if wandb.run is None:
                return

            metrics = {f"{prefix}/accuracy": accuracy["overall"]}
            for topic, info in accuracy["per_topic"].items():
                metrics[f"{prefix}/acc_{topic}"] = info["acc"]

            # Log difficulty breakdown if available
            if "per_difficulty" in accuracy:
                for diff, info in accuracy["per_difficulty"].items():
                    metrics[f"{prefix}/acc_{diff}"] = info["acc"]

            # Delta from baseline
            if self.baseline:
                delta = accuracy["overall"] - self.baseline.get("overall", 0)
                metrics[f"{prefix}/delta_vs_base"] = delta

            wandb.log(metrics, step=step)
        except Exception as e:
            print(f"[wandb] Error logging accuracy: {e}")

    def _write_verdict(self, accuracy: dict, epoch: int, step: int, elapsed: float, path: Path):
        """Write epoch verdict markdown."""
        lines = [
            f"# Epoch {epoch} Verdict\n",
            f"**Step:** {step} | **Eval time:** {elapsed:.0f}s | **Samples:** {accuracy['n_total']}\n",
            f"## Overall: {accuracy['overall']:.1%} ({accuracy['n_correct']}/{accuracy['n_total']})\n",
        ]

        # Baseline comparison
        if self.baseline:
            base_acc = self.baseline.get("overall", 0)
            delta = accuracy["overall"] - base_acc
            lines.append(f"**vs Base model:** {base_acc:.1%} → {accuracy['overall']:.1%} "
                         f"(**{'+' if delta >= 0 else ''}{delta:.1%}**)\n")

        # Per-topic table
        lines.append("## Per-Topic Accuracy\n")
        lines.append("| Topic | Accuracy | Correct/Total | vs Base |")
        lines.append("|-------|----------|---------------|---------|")
        for topic, info in sorted(accuracy["per_topic"].items()):
            base_topic_acc = self.baseline.get("per_topic", {}).get(topic, {}).get("acc", 0) if self.baseline else 0
            delta_t = info["acc"] - base_topic_acc
            lines.append(f"| {topic} | {info['acc']:.1%} | {info['correct']}/{info['n']} | "
                         f"{'+' if delta_t >= 0 else ''}{delta_t:.1%} |")

        # Trend (quick eval history)
        if len(self.quick_history) > 1:
            lines.append("\n## Quick Eval Trend\n")
            lines.append("| Step | Epoch | Accuracy |")
            lines.append("|------|-------|----------|")
            for h in self.quick_history:
                lines.append(f"| {h['step']} | {h['epoch']:.2f} | {h['overall']:.1%} |")

        # Full eval history
        if len(self.full_history) > 1:
            lines.append("\n## Full Eval History\n")
            lines.append("| Epoch | Accuracy | Delta |")
            lines.append("|-------|----------|-------|")
            prev_acc = self.baseline.get("overall", 0) if self.baseline else 0
            for h in self.full_history:
                d = h["overall"] - prev_acc
                lines.append(f"| {h['epoch']} | {h['overall']:.1%} | {'+' if d >= 0 else ''}{d:.1%} |")
                prev_acc = h["overall"]

        # Overfitting check
        if len(self.full_history) >= 2:
            prev = self.full_history[-2]["overall"]
            curr = self.full_history[-1]["overall"]
            if curr < prev - 0.02:
                lines.append(f"\n**WARNING: Accuracy dropped {prev:.1%} → {curr:.1%}. Possible overfitting.**\n")
            elif curr > prev:
                lines.append(f"\n**Model still improving.** (+{curr - prev:.1%} from previous epoch)\n")
            else:
                lines.append(f"\n**Plateauing.** ({prev:.1%} → {curr:.1%})\n")

        with open(path, "w") as f:
            f.write("\n".join(lines))
        print(f"[eval] Verdict written to {path}")

    def _upload_verdict(self, path: Path, epoch: int):
        """Upload verdict file to W&B as artifact."""
        try:
            import wandb
            if wandb.run is None:
                return
            artifact = wandb.Artifact(f"epoch_{epoch}_verdict", type="report")
            artifact.add_file(str(path))
            wandb.log_artifact(artifact)

            # Also log as text to W&B for inline viewing
            with open(path) as f:
                content = f.read()
            wandb.log({f"verdict/epoch_{epoch}": wandb.Html(f"<pre>{content}</pre>")})
        except Exception as e:
            print(f"[wandb] Error uploading verdict: {e}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_training_curves(
    loss_logger: LossLogger,
    eval_callback: EvalCallback | None,
    run_dir: Path,
    profile_name: str,
):
    """Generate comprehensive training curves."""
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
    has_acc = eval_callback and len(eval_callback.quick_history) > 0

    n_plots = 2 + int(has_eval) + int(has_acc)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

    ax_idx = 0

    # 1. Train + eval loss
    axes[ax_idx].plot(loss_logger.steps, loss_logger.losses, "b-", linewidth=1, alpha=0.6, label="Train")
    if has_eval:
        axes[ax_idx].plot(loss_logger.eval_steps, loss_logger.eval_losses, "ro-",
                         markersize=5, linewidth=2, label="Val Loss")
    axes[ax_idx].axhline(y=1.386, color="gray", linestyle="--", alpha=0.3, label="Random")
    axes[ax_idx].set_xlabel("Step")
    axes[ax_idx].set_ylabel("Loss")
    axes[ax_idx].set_title("Loss")
    axes[ax_idx].legend(fontsize=9)
    axes[ax_idx].grid(True, alpha=0.3)
    ax_idx += 1

    # 2. LR schedule
    axes[ax_idx].plot(loss_logger.steps, loss_logger.lrs, "r-", linewidth=1.5)
    axes[ax_idx].set_xlabel("Step")
    axes[ax_idx].set_ylabel("Learning Rate")
    axes[ax_idx].set_title("LR Schedule")
    axes[ax_idx].grid(True, alpha=0.3)
    ax_idx += 1

    # 3. Accuracy over time (quick + full)
    if has_acc:
        q_steps = [h["step"] for h in eval_callback.quick_history]
        q_accs = [h["overall"] for h in eval_callback.quick_history]
        axes[ax_idx].plot(q_steps, q_accs, "gs-", markersize=5, linewidth=1.5, label="Quick (100)")

        if eval_callback.full_history:
            f_steps = [h["step"] for h in eval_callback.full_history]
            f_accs = [h["overall"] for h in eval_callback.full_history]
            axes[ax_idx].plot(f_steps, f_accs, "rD-", markersize=7, linewidth=2, label="Full (all val)")

        if eval_callback.baseline:
            axes[ax_idx].axhline(y=eval_callback.baseline["overall"], color="gray",
                                linestyle="--", alpha=0.5, label=f"Base ({eval_callback.baseline['overall']:.0%})")
            axes[ax_idx].axhline(y=0.25, color="orange", linestyle=":", alpha=0.4, label="Random (25%)")

        axes[ax_idx].set_xlabel("Step")
        axes[ax_idx].set_ylabel("Accuracy")
        axes[ax_idx].set_title("Accuracy")
        axes[ax_idx].legend(fontsize=9)
        axes[ax_idx].set_ylim(0, 1)
        axes[ax_idx].grid(True, alpha=0.3)
        ax_idx += 1

    # 4. Grad norms (if available)
    if has_eval and loss_logger.grad_norms and any(g > 0 for g in loss_logger.grad_norms):
        axes[ax_idx].plot(loss_logger.steps, loss_logger.grad_norms, "m-", linewidth=0.8, alpha=0.6)
        axes[ax_idx].set_xlabel("Step")
        axes[ax_idx].set_ylabel("Grad Norm")
        axes[ax_idx].set_title("Gradient Norms")
        axes[ax_idx].grid(True, alpha=0.3)

    fig.suptitle(f"EOLLM Vision SFT — Qwen3.5-4B LoRA ({profile_name})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_path = run_dir / "training_curves.png"
    fig.savefig(str(plot_path), dpi=150)
    plt.close()
    print(f"Training curves saved to {plot_path}")

    # Upload to W&B
    try:
        import wandb
        if wandb.run:
            wandb.log({"training_curves": wandb.Image(str(plot_path))})
    except Exception:
        pass
