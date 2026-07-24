"""
Central configuration for the depth-based navigation system.
Tune these values based on real-world testing with your specific
camera, chassis, and environment - the numbers below are reasonable
starting points, not guaranteed-correct for your rover.
"""
import logging
from pathlib import Path


class Config:
    # ---- Paths ----
    MODEL_PATH = Path(__file__).parent / "models" / "midas_small_ts.pt"

    # ---- Camera ----
    CAMERA_INDEX = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480

    # ---- Model input ----
    MODEL_INPUT_SIZE = 256  # MiDaS_small's expected input resolution

    # ---- Zone analysis (fraction of frame height to examine) ----
    FOCUS_TOP_RATIO = 0.3     # ignore top 30% (sky / ceiling)
    FOCUS_BOTTOM_RATIO = 0.9  # ignore bottom 10% (ground right under the rover)

    # ---- Decision thresholds (0-1 normalized "closeness") ----
    DANGER_THRESHOLD = 0.65     # start reacting to an obstacle
    CRITICAL_THRESHOLD = 0.85   # obstacle is basically in front of us -> stop
    MAX_STEERING_ANGLE = 60.0   # degrees, physical steering limit
    SENSITIVITY = 150.0         # base gain for steering-angle calculation

    # ---- Speed control ----
    BASE_SPEED = 0.3   # 0.0 - 1.0, normal cruising speed
    MIN_SPEED = 0.1    # never crawl slower than this while still moving

    # ---- Loop / performance ----
    TARGET_FPS = 10     # throttle inference rate (RPi CPUs can't do MiDaS at full speed)
    TORCH_THREADS = 4   # tune to your Pi's core count (4 for Pi 4/5)

    # ---- Runtime behavior ----
    HEADLESS = True   # True on the actual rover (no monitor attached)
    LOG_LEVEL = logging.INFO


def get_logger(name):
    logging.basicConfig(
        level=Config.LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return logging.getLogger(name)