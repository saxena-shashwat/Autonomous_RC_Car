"""
Production autonomous navigation loop for a Raspberry Pi rover.

Pipeline: camera -> sliding-window compression -> MiDaS depth estimation ->
zone-based obstacle analysis -> steering/speed decision -> motor control,
running continuously until interrupted.

Setup (see README.md for the full walkthrough):
    1. On a machine WITH internet, run: python export_model.py
       This downloads MiDaS_small once and saves it as an offline
       TorchScript file under models/midas_small_ts.pt
    2. Copy the whole rover_nav/ folder (including models/) to the Pi.
    3. On the Pi:  pip install -r requirements.txt
    4. Edit motor_controller.py to match your actual wiring.
    5. Run:  python navigation_car.py            (headless, for the actual rover)
             python navigation_car.py --show      (with debug windows, needs a display)
"""
import argparse
import signal
import time
from collections import deque

import cv2
import numpy as np
import torch
from threading import Thread, Lock

from config import Config, get_logger
from motor_controller import MotorController

logger = get_logger("nav")


# ---------------------------------------------------------------------------
# Threaded camera reader: grabs frames continuously in the background so the
# main loop always processes the freshest frame instead of blocking on I/O.
# ---------------------------------------------------------------------------
class CameraStream:
    def __init__(self, index=0, width=640, height=480):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize latency

        self.lock = Lock()
        self.frame = None
        self.ok = False
        self.stopped = False

        for _ in range(5):  # warm up / let auto-exposure settle
            self.cap.read()

        self.thread = Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            with self.lock:
                self.ok = ok
                if ok:
                    self.frame = frame
            if not ok:
                time.sleep(0.05)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ok, self.frame.copy()

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=1)
        self.cap.release()


# ---------------------------------------------------------------------------
# Sliding-window frame compressor: rolling average over the last N frames,
# same averaging logic as the offline video_compression.py script but
# applied frame-by-frame in real time instead of over a whole array at once.
# ---------------------------------------------------------------------------
class SlidingWindowCompressor:
    def __init__(self, window_size=6):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def add_and_get(self, frame):
        self.buffer.append(frame.astype(np.float64))
        if len(self.buffer) < self.window_size:
            return frame  # not enough history yet, pass raw frame
        avg = np.sum(self.buffer, axis=0) / self.window_size
        return avg.astype(np.uint8)


# ---------------------------------------------------------------------------
# Depth model wrapper - loads the pre-exported TorchScript file, no internet
# or torch.hub calls needed at runtime.
# ---------------------------------------------------------------------------
class DepthModel:
    def __init__(self, model_path, input_size=256, num_threads=4):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {model_path}.\n"
                f"Run export_model.py first (on a machine with internet access) "
                f"to generate it, then copy it here."
            )
        logger.info(f"Loading cached model from {model_path} ...")
        self.device = torch.device("cpu")  # Raspberry Pi has no CUDA
        torch.set_num_threads(num_threads)

        self.model = torch.jit.load(str(model_path), map_location=self.device)
        self.model.eval()

        self.input_size = input_size
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self._warmup()
        logger.info("Model loaded and warmed up.")

    def _preprocess(self, frame_bgr):
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        tensor = (tensor - self.mean) / self.std
        return tensor.to(self.device)

    def _warmup(self):
        dummy = torch.zeros(1, 3, self.input_size, self.input_size)
        with torch.no_grad():
            self.model(dummy)

    @torch.no_grad()
    def infer(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        inp = self._preprocess(frame_bgr)
        pred = self.model(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
        ).squeeze()
        depth = pred.cpu().numpy()
        return cv2.normalize(depth, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


# ---------------------------------------------------------------------------
# Navigation decision logic - this is your "dynamic turning" version
# (90th-percentile zones + closeness-squared sensitivity), plus an added
# emergency-stop layer and speed scaling that your original code didn't have.
# ---------------------------------------------------------------------------
class Navigator:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def decide(self, depth_norm):
        cfg = self.cfg
        h, w = depth_norm.shape
        third = w // 3
        top = int(h * cfg.FOCUS_TOP_RATIO)
        bottom = int(h * cfg.FOCUS_BOTTOM_RATIO)

        left_zone = depth_norm[top:bottom, 0:third]
        center_zone = depth_norm[top:bottom, third:2 * third]
        right_zone = depth_norm[top:bottom, 2 * third:w]

        left_c = float(np.percentile(left_zone, 90))
        center_c = float(np.percentile(center_zone, 90))
        right_c = float(np.percentile(right_zone, 90))
        closest = max(left_c, center_c, right_c)

        result = {
            "left": left_c, "center": center_c, "right": right_c,
            "closest": closest, "steering_angle": 0.0,
            "speed": cfg.BASE_SPEED, "status": "CLEAR",
        }

        # Emergency stop: something is right in front of us no matter which zone
        if closest >= cfg.CRITICAL_THRESHOLD:
            result.update(speed=0.0, steering_angle=0.0, status="EMERGENCY_STOP")
            return result

        if closest < cfg.DANGER_THRESHOLD:
            return result  # CLEAR, defaults are fine

        # Reacting to an obstacle: sensitivity scales with how close it is
        # (gentle reaction far away, sharp reaction close up)
        dynamic_sensitivity = cfg.SENSITIVITY * (closest ** 2)
        imbalance = left_c - right_c
        angle = float(np.clip(imbalance * dynamic_sensitivity,
                               -cfg.MAX_STEERING_ANGLE, cfg.MAX_STEERING_ANGLE))

        # Slow down the closer/sharper the turn - your original code computed
        # base_speed but never actually used it anywhere. This fixes that.
        speed_scale = max(0.0, 1.0 - (closest - cfg.DANGER_THRESHOLD) / (1.0 - cfg.DANGER_THRESHOLD))
        speed = max(cfg.MIN_SPEED, cfg.BASE_SPEED * speed_scale)

        result.update(
            steering_angle=angle,
            speed=speed,
            status="TURN_RIGHT" if angle > 0 else "TURN_LEFT",
        )
        return result


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class App:
    def __init__(self, args):
        self.cfg = Config()
        self.cfg.HEADLESS = not args.show
        self.cfg.CAMERA_INDEX = args.camera

        self.camera = CameraStream(self.cfg.CAMERA_INDEX, self.cfg.FRAME_WIDTH, self.cfg.FRAME_HEIGHT)
        self.compressor = SlidingWindowCompressor(window_size=args.window_size)
        self.model = DepthModel(self.cfg.MODEL_PATH, self.cfg.MODEL_INPUT_SIZE, self.cfg.TORCH_THREADS)
        self.nav = Navigator(self.cfg)
        self.motors = MotorController()
        self.running = True

        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        logger.info("Shutdown signal received - stopping motors and exiting...")
        self.running = False

    def run(self):
        frame_interval = 1.0 / self.cfg.TARGET_FPS
        consecutive_failures = 0

        try:
            while self.running:
                loop_start = time.time()
                ok, frame = self.camera.read()

                if not ok or frame is None:
                    consecutive_failures += 1
                    logger.warning("No camera frame available, retrying...")
                    if consecutive_failures >= 20:
                        # Camera has been down for ~a couple seconds straight - play it safe.
                        self.motors.stop()
                    time.sleep(0.1)
                    continue
                consecutive_failures = 0

                frame = self.compressor.add_and_get(frame)

                try:
                    depth = self.model.infer(frame)
                except Exception as e:
                    logger.error(f"Inference failed: {e}")
                    self.motors.stop()  # fail safe: stop rather than drive blind
                    time.sleep(0.2)
                    continue

                decision = self.nav.decide(depth)
                self.motors.set_steering(decision["steering_angle"])
                self.motors.set_speed(decision["speed"])

                logger.info(
                    f"L={decision['left']:.2f} C={decision['center']:.2f} R={decision['right']:.2f} "
                    f"-> {decision['status']} angle={decision['steering_angle']:.1f} "
                    f"speed={decision['speed']:.2f}"
                )

                if not self.cfg.HEADLESS:
                    self._show_debug(frame, depth, decision)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                elapsed = time.time() - loop_start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
        finally:
            self.shutdown()

    def _show_debug(self, frame, depth, decision):
        depth_vis = cv2.applyColorMap(np.uint8(depth * 255), cv2.COLORMAP_JET)
        text = f"{decision['status']} {decision['steering_angle']:.1f}deg spd={decision['speed']:.2f}"
        color = (0, 255, 0) if decision["status"] == "CLEAR" else (0, 0, 255)
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Camera", frame)
        cv2.imshow("Depth", depth_vis)

    def shutdown(self):
        logger.info("Cleaning up...")
        self.motors.stop()
        self.camera.stop()
        cv2.destroyAllWindows()
        logger.info("Shutdown complete.")


def parse_args():
    p = argparse.ArgumentParser(description="Depth-based autonomous navigation for a Raspberry Pi rover")
    p.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    p.add_argument("--show", action="store_true", help="Show debug windows (requires a connected display)")
    p.add_argument("--window-size", type=int, default=6, help="Sliding-window size for frame compression (default: 6)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    App(args).run()