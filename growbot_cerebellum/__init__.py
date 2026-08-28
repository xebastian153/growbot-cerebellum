"""growbot-cerebellum: the library behind the experiment scripts at the repository root.

    sim         the MuJoCo twin at 50 Hz, ServoModel, perturb(), collect()
    forward     features, forward models, make_windows, rollout_error, by_regime, K
    sim2real    horizon_within, the online residual, the 13 DR corners
    servo_id    actuator identification from IMU + commands through the frozen model
    sensor_id   fusion-filter lag, Allan deviation, dt statistics
    imulog      the ?imulog=1 parser, preflight, segmenter and fixture
    gap         evaluate_axes, twin_regimes, REGIME_MAP
    honesty     seed_stat, per-metric bars, resolved verdicts, score_corners
    planner     Imagination and the CEM planner
    tee         Tee, the run-log mirror
    provenance  provenance(), the block every script writes into its artifact (older artifacts predate it)
    paths       ROOT, DATA, RESULTS, LOGS — the scripts read and write through these, from any cwd

Every script at the root is a thin CLI over these modules; every documented command still
runs as written.
"""
from .provenance import provenance

__version__ = "0.1.0"
__all__ = ["provenance", "__version__"]
