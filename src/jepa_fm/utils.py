import math

import mlflow


class MetricLog:
    """Buffer step-indexed metrics and push them to MLflow in batches.

    Every point is also kept in `history`, which the end-of-run figures use.
    """

    def __init__(self, flush_every: int = 50, max_pending: int = 200):
        self.flush_every = flush_every
        self.max_pending = max_pending
        self.pending: dict[int, dict[str, float]] = {}
        self.history: dict[str, dict[int, float]] = {}

    def log(self, metrics: dict[str, float], step: int) -> None:
        self.pending[step] = metrics
        for key, value in metrics.items():
            self.history.setdefault(key, {})[step] = value
        if len(self.pending) > self.max_pending:  # mlflow server is down
            for stale in sorted(self.pending)[: len(self.pending) - self.max_pending]:
                del self.pending[stale]
        if len(self.pending) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        try:
            for step in sorted(self.pending):
                mlflow.log_metrics(self.pending[step], step=step)
        except Exception as exc:
            print(f"mlflow flush failed, {len(self.pending)} points kept: {exc}")
            return
        self.pending.clear()


def log_metric_figures(history: dict[str, dict[int, float]]) -> list[str]:
    """Draw one PNG per metric prefix (train/, val/, probe/) as a run artifact."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # no display on the training box
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed, skipping metric figures")
        return []

    groups: dict[str, dict[str, dict[int, float]]] = {}
    for key, points in history.items():
        prefix, _, name = key.partition("/")
        groups.setdefault(prefix, {})[name or prefix] = points

    logged: list[str] = []
    for group, series in sorted(groups.items()):
        # a subplot per metric: loss and a learning rate cannot share a y axis
        rows = math.ceil(len(series) / 2)
        fig, axes = plt.subplots(rows, 2, figsize=(10, 2.8 * rows), squeeze=False)
        for ax, (name, points) in zip(axes.flat, sorted(series.items())):
            steps = sorted(points)
            ax.plot(steps, [points[step] for step in steps], linewidth=1.4)
            ax.set_title(name, fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)
        for ax in list(axes.flat)[len(series) :]:
            ax.axis("off")
        fig.suptitle(group, fontsize=11)
        fig.tight_layout()

        artifact = f"figures/{group}.png"
        mlflow.log_figure(fig, artifact)
        plt.close(fig)
        logged.append(artifact)
    return logged
