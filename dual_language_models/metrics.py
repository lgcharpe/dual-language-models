"""Training metrics tracking for throughput and compute efficiency.

Provides TrainingMetrics for tracking tokens/sec, step timing, FLOPs
estimation, MFU, and memory usage. Produces an efficiency summary at
the end of training.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch


@dataclass
class StepMetrics:
    """Metrics captured for a single training step."""

    tokens_per_sec: float = 0.0
    samples_per_sec: float = 0.0
    step_time_ms: float = 0.0
    flops_per_step: float = 0.0
    mfu: float = 0.0
    memory_allocated_mb: float = 0.0
    memory_reserved_mb: float = 0.0


class TrainingMetrics:
    """Tracks throughput, compute efficiency, and hardware utilization.

    Args:
        num_params: Total trainable parameters in the model.
        seq_len: Sequence length per sample.
        world_size: Number of distributed processes.
        device: Training device (for memory tracking).
        peak_flops_per_sec: Theoretical peak FLOPS of the device (for MFU).
            If None, MFU is not computed. Common values:
            - A100 SXM bf16: 312e12
            - H100 SXM bf16: 989e12
            - Apple M2 Ultra: ~27e12
    """

    def __init__(
        self,
        num_params: int,
        seq_len: int,
        world_size: int = 1,
        device: torch.device | str = "cpu",
        peak_flops_per_sec: float | None = None,
    ):
        self.num_params = num_params
        self.seq_len = seq_len
        self.world_size = world_size
        self.device = torch.device(device) if isinstance(device, str) else device
        self.peak_flops_per_sec = peak_flops_per_sec

        # Running state
        self._step_start: float = 0.0
        self._total_tokens: int = 0
        self._total_steps: int = 0
        self._total_time: float = 0.0
        self._peak_memory_mb: float = 0.0
        self._training_start: float = time.time()

        # Smoothed metrics (EMA)
        self._ema_tokens_per_sec: float = 0.0
        self._ema_step_time: float = 0.0
        self._ema_alpha: float = 0.05

    def step_start(self) -> None:
        """Mark the beginning of a training step."""
        self._step_start = time.time()

    def step_end(self, batch_size: int) -> StepMetrics:
        """Mark the end of a training step and return computed metrics.

        Args:
            batch_size: Global batch size for this step.

        Returns:
            StepMetrics with throughput and efficiency numbers.
        """
        elapsed = time.time() - self._step_start
        tokens_this_step = batch_size * self.seq_len

        self._total_tokens += tokens_this_step
        self._total_steps += 1
        self._total_time += elapsed

        tokens_per_sec = tokens_this_step / elapsed if elapsed > 0 else 0.0
        samples_per_sec = batch_size / elapsed if elapsed > 0 else 0.0

        # EMA smoothing
        if self._total_steps == 1:
            self._ema_tokens_per_sec = tokens_per_sec
            self._ema_step_time = elapsed
        else:
            self._ema_tokens_per_sec = (1 - self._ema_alpha) * self._ema_tokens_per_sec + self._ema_alpha * tokens_per_sec
            self._ema_step_time = (1 - self._ema_alpha) * self._ema_step_time + self._ema_alpha * elapsed

        # FLOPs estimate: ~6 * num_params * tokens for a transformer forward+backward
        flops_per_step = 6.0 * self.num_params * tokens_this_step

        # MFU: fraction of theoretical peak utilized
        mfu = 0.0
        if self.peak_flops_per_sec is not None and elapsed > 0:
            mfu = flops_per_step / (elapsed * self.peak_flops_per_sec * self.world_size)

        # Memory tracking
        memory_allocated_mb = 0.0
        memory_reserved_mb = 0.0
        if self.device.type == "cuda" and torch.cuda.is_available():
            memory_allocated_mb = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
            memory_reserved_mb = torch.cuda.memory_reserved(self.device) / (1024 * 1024)
            peak = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
            self._peak_memory_mb = max(self._peak_memory_mb, peak)
        elif self.device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
            memory_allocated_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
            self._peak_memory_mb = max(self._peak_memory_mb, memory_allocated_mb)

        return StepMetrics(
            tokens_per_sec=tokens_per_sec,
            samples_per_sec=samples_per_sec,
            step_time_ms=elapsed * 1000.0,
            flops_per_step=flops_per_step,
            mfu=mfu,
            memory_allocated_mb=memory_allocated_mb,
            memory_reserved_mb=memory_reserved_mb,
        )

    def get_smoothed(self) -> dict[str, float]:
        """Return EMA-smoothed metrics for logging."""
        return {
            "throughput/tokens_per_sec": self._ema_tokens_per_sec,
            "throughput/tokens_per_sec_total": self._ema_tokens_per_sec * self.world_size,
            "throughput/step_time_ms": self._ema_step_time * 1000.0,
        }

    def summary(self) -> str:
        """Return an efficiency summary string for end-of-training reporting."""
        wall_time = time.time() - self._training_start
        avg_tokens_per_sec = self._total_tokens / self._total_time if self._total_time > 0 else 0.0

        lines = [
            "",
            "=" * 60,
            "  TRAINING EFFICIENCY SUMMARY",
            "=" * 60,
            f"  Total tokens trained:     {self._total_tokens:,.0f}",
            f"  Total training steps:     {self._total_steps:,}",
            f"  Total wall-clock time:    {wall_time / 3600:.2f} hours",
            f"  Active training time:     {self._total_time / 3600:.2f} hours",
            f"  Avg tokens/sec (device):  {avg_tokens_per_sec:,.0f}",
            f"  Avg tokens/sec (total):   {avg_tokens_per_sec * self.world_size:,.0f}",
            f"  Avg step time:            {self._total_time / max(self._total_steps, 1) * 1000:.1f} ms",
            f"  Peak memory (MB):         {self._peak_memory_mb:.1f}",
            f"  Model parameters:         {self.num_params:,}",
        ]
        if self.peak_flops_per_sec is not None and self._total_time > 0:
            total_flops = 6.0 * self.num_params * self._total_tokens
            avg_mfu = total_flops / (self._total_time * self.peak_flops_per_sec * self.world_size)
            lines.append(f"  Average MFU:              {avg_mfu * 100:.1f}%")
        lines.append("=" * 60)
        return "\n".join(lines)
