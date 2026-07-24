#!/usr/bin/env python3
"""
RL & ML Vision Interface Node for Vision-Based Autonomous Robot

Subscribes to camera images (/camera/image_raw), runs MiDaS ML depth estimation and
zone-based obstacle avoidance navigation, and publishes velocity commands (/cmd_vel).
"""

import os
import sys
import math
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Union, Dict, Any

import cv2
import numpy as np

# Try importing ROS2 dependencies
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import Image
    from geometry_msgs.msg import Twist
    from std_msgs.msg import String, Float32MultiArray
    from cv_bridge import CvBridge
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    # Dummy Twist class for non-ROS standalone testing
    class Twist:
        def __init__(self):
            class Vector3:
                def __init__(self):
                    self.x = 0.0
                    self.y = 0.0
                    self.z = 0.0
            self.linear = Vector3()
            self.angular = Vector3()

# Try importing torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class RLConfig:
    """Configuration for RL & ML Vision node behavior."""
    model_path: str = ""
    action_type: str = "continuous"  # "continuous" or "discrete"
    inference_rate: float = 20.0  # Hz
    max_linear_speed: float = 2.0  # m/s
    max_angular_speed: float = 4.0  # rad/s
    discrete_linear: float = 1.0  # m/s for discrete FORWARD
    discrete_angular: float = 2.0  # rad/s for discrete LEFT/RIGHT
    image_width: int = 640
    image_height: int = 480
    model_input_size: int = 256
    focus_top_ratio: float = 0.3
    focus_bottom_ratio: float = 0.9
    danger_threshold: float = 0.65
    critical_threshold: float = 0.85
    max_steering_angle: float = 60.0
    sensitivity: float = 150.0
    base_speed: float = 0.5
    min_speed: float = 0.1
    use_gpu: bool = True
    debug_publish: bool = True


class SlidingWindowCompressor:
    """Rolling average over last N frames to smooth camera input noise."""
    def __init__(self, window_size: int = 4):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def add_and_get(self, frame: np.ndarray) -> np.ndarray:
        if self.window_size <= 1:
            return frame
        self.buffer.append(frame.astype(np.float64))
        if len(self.buffer) < self.window_size:
            return frame
        avg = np.sum(self.buffer, axis=0) / len(self.buffer)
        return avg.astype(np.uint8)


class Navigator:
    """Zone-based obstacle avoidance decision engine based on depth map."""
    def __init__(self, config: RLConfig):
        self.config = config

    def decide(self, depth_norm: np.ndarray) -> Dict[str, Any]:
        cfg = self.config
        h, w = depth_norm.shape
        third = w // 3
        top = int(h * cfg.focus_top_ratio)
        bottom = int(h * cfg.focus_bottom_ratio)

        left_zone = depth_norm[top:bottom, 0:third]
        center_zone = depth_norm[top:bottom, third:2 * third]
        right_zone = depth_norm[top:bottom, 2 * third:w]

        left_c = float(np.percentile(left_zone, 90)) if left_zone.size > 0 else 0.0
        center_c = float(np.percentile(center_zone, 90)) if center_zone.size > 0 else 0.0
        right_c = float(np.percentile(right_zone, 90)) if right_zone.size > 0 else 0.0
        closest = max(left_c, center_c, right_c)

        result = {
            "left": left_c,
            "center": center_c,
            "right": right_c,
            "closest": closest,
            "steering_angle": 0.0,
            "speed": cfg.base_speed,
            "status": "CLEAR",
        }

        # Emergency stop threshold
        if closest >= cfg.critical_threshold:
            result.update(speed=0.0, steering_angle=0.0, status="EMERGENCY_STOP")
            return result

        if closest < cfg.danger_threshold:
            return result  # Path clear

        # Reacting to an obstacle: sensitivity scales quadratic with closeness
        dynamic_sensitivity = cfg.sensitivity * (closest ** 2)
        imbalance = left_c - right_c
        angle = float(np.clip(imbalance * dynamic_sensitivity, -cfg.max_steering_angle, cfg.max_steering_angle))

        # Speed scaling: decrease speed as obstacle gets closer
        speed_range = max(1e-5, 1.0 - cfg.danger_threshold)
        speed_scale = max(0.0, 1.0 - (closest - cfg.danger_threshold) / speed_range)
        speed = max(cfg.min_speed, cfg.base_speed * speed_scale)

        result.update(
            steering_angle=angle,
            speed=speed,
            status="TURN_RIGHT" if angle > 0 else "TURN_LEFT",
        )
        return result


class RLModelWrapper:
    """
    ML Depth Model Wrapper & Inference Engine.

    Loads the MiDaS depth estimation model (TorchScript or PyTorch Hub),
    preprocesses ROS camera frames, runs neural network inference,
    computes zone depth analysis via Navigator, and converts output to ROS Twist.
    """

    def __init__(self, config: RLConfig, logger=None):
        self.config = config
        self.logger = logger
        self.model = None
        self.device = None
        self.is_heuristic = False
        self.navigator = Navigator(config)
        self.compressor = SlidingWindowCompressor(window_size=4)

        if TORCH_AVAILABLE:
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self._load_model()

    def _log(self, msg: str, level: str = "info"):
        if self.logger:
            getattr(self.logger, level)(msg)
        else:
            print(f"[RLModelWrapper] {msg}")

    def _load_model(self):
        """Load MiDaS depth model with robust fallback handling."""
        if not TORCH_AVAILABLE:
            self._log("PyTorch is not available. Running in Heuristic Vision Mode.", "warning")
            self.is_heuristic = True
            return

        self.device = torch.device("cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu")
        self._log(f"Using compute device: {self.device}")

        # Add torch hub cache directory to sys.path if present
        hub_dir = torch.hub.get_dir()
        for folder in os.listdir(hub_dir) if os.path.exists(hub_dir) else []:
            folder_path = os.path.join(hub_dir, folder)
            if os.path.isdir(folder_path) and folder_path not in sys.path:
                sys.path.insert(0, folder_path)

        # Search paths for cached model
        search_paths = []
        if self.config.model_path:
            search_paths.append(self.config.model_path)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        search_paths.extend([
            os.path.join(base_dir, "..", "models", "midas_small.pt"),
            os.path.join(base_dir, "..", "models", "midas_small_weights.pt"),
            os.path.join(base_dir, "..", "models", "midas_small_ts.pt"),
            os.path.join(os.getcwd(), "production_code", "models", "midas_small.pt"),
            os.path.join(os.getcwd(), "production_code", "models", "midas_small_weights.pt"),
            os.path.join(os.getcwd(), "production_code", "models", "midas_small_ts.pt"),
            os.path.join(os.getcwd(), "src", "robot_controller", "models", "midas_small.pt"),
        ])

        model_loaded = False
        for path in search_paths:
            path = os.path.abspath(path)
            if os.path.exists(path):
                try:
                    self._log(f"Loading cached depth model from {path}...")
                    if path.endswith("_ts.pt"):
                        self.model = torch.jit.load(path, map_location=self.device)
                    elif path.endswith("_weights.pt"):
                        arch_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True, skip_validation=True)
                        arch_model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
                        self.model = arch_model
                    else:
                        self.model = torch.load(path, map_location=self.device, weights_only=False)
                    self.model.to(self.device)
                    self.model.eval()
                    model_loaded = True
                    self._log(f"Successfully loaded model from {path}!")
                    break
                except Exception as e:
                    self._log(f"Failed to load model from {path}: {e}", "warning")

        if not model_loaded:
            try:
                self._log("Cached model file not found. Attempting PyTorch Hub download (MiDaS_small)...")
                self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True, skip_validation=True)
                self.model.to(self.device)
                self.model.eval()
                model_loaded = True
                self._log("Successfully loaded MiDaS_small from PyTorch Hub!")
            except Exception as e:
                self._log(f"PyTorch Hub model load failed: {e}", "warning")

        if not model_loaded:
            self._log("⚠️ No ML model loaded. Running in Heuristic Vision Depth Mode.", "warning")
            self.is_heuristic = True

    def preprocess(self, cv_image: np.ndarray) -> Any:
        """Preprocess OpenCV BGR image into model tensor input."""
        smooth_frame = self.compressor.add_and_get(cv_image)

        if self.is_heuristic or not TORCH_AVAILABLE:
            return smooth_frame

        img = cv2.cvtColor(smooth_frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.config.model_input_size, self.config.model_input_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        tensor = (tensor - self.mean) / self.std
        return tensor.to(self.device)

    def predict(self, preprocessed_input: Any) -> Dict[str, Any]:
        """
        Run depth estimation model inference and obstacle navigation decision.

        Returns decision dictionary containing steering_angle, speed, status, and zone closeness values.
        """
        if self.is_heuristic or not TORCH_AVAILABLE or self.model is None:
            frame = preprocessed_input if isinstance(preprocessed_input, np.ndarray) else np.zeros((480, 640, 3), dtype=np.uint8)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            depth_map = 1.0 - (gray.astype(np.float32) / 255.0)
            return self.navigator.decide(depth_map)

        h, w = self.config.image_height, self.config.image_width
        with torch.no_grad():
            pred = self.model(preprocessed_input)
            if isinstance(pred, tuple):
                pred = pred[0]
            if pred.ndim == 3:
                pred = pred.unsqueeze(1)
            pred = torch.nn.functional.interpolate(
                pred, size=(h, w), mode="bicubic", align_corners=False
            ).squeeze()
            depth = pred.cpu().numpy()

        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max > depth_min:
            depth_norm = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = np.zeros_like(depth)

        return self.navigator.decide(depth_norm)

    def action_to_twist(self, decision: Union[Dict[str, Any], np.ndarray, int]) -> Twist:
        """
        Convert navigation decision to ROS2 Twist velocity command.
        """
        twist = Twist()

        if isinstance(decision, dict):
            speed = float(decision.get("speed", self.config.base_speed))
            steering_angle = float(decision.get("steering_angle", 0.0))

            # ROS frame: positive linear.x = forward, positive angular.z = turn left (counter-clockwise)
            # If steering_angle > 0 (turn right), angular.z should be negative.
            linear = float(np.clip(speed, 0.0, self.config.max_linear_speed))
            angular = float(np.clip(- (steering_angle / self.config.max_steering_angle) * self.config.max_angular_speed,
                                     -self.config.max_angular_speed, self.config.max_angular_speed))

            twist.linear.x = linear
            twist.angular.z = angular
            return twist

        if self.config.action_type == "continuous":
            if isinstance(decision, (list, tuple, np.ndarray)):
                linear = float(np.clip(decision[0], -self.config.max_linear_speed, self.config.max_linear_speed))
                angular = float(np.clip(decision[1], -self.config.max_angular_speed, self.config.max_angular_speed))
            else:
                linear = angular = 0.0
            twist.linear.x = linear
            twist.angular.z = angular
        elif self.config.action_type == "discrete":
            action_map = {
                0: (0.0, self.config.discrete_angular),      # LEFT
                1: (0.0, -self.config.discrete_angular),     # RIGHT
                2: (self.config.discrete_linear, 0.0),        # FORWARD
            }
            action_idx = int(decision) if decision is not None else 2
            linear, angular = action_map.get(action_idx, (0.0, 0.0))
            twist.linear.x = linear
            twist.angular.z = angular

        return twist


if ROS_AVAILABLE:
    class RLNode(Node):
        """
        ROS2 Node bridging camera images -> ML MiDaS Depth Inference -> ROS2 Twist Velocity Commands.
        """

        def __init__(self):
            super().__init__('rl_node')

            # Declare parameters safely
            params_to_declare = [
                ('model_path', ''),
                ('camera_topic', '/camera/image_raw'),
                ('cmd_vel_topic', '/cmd_vel'),
                ('inference_rate', 20.0),
                ('action_type', 'continuous'),
                ('max_linear_speed', 2.0),
                ('max_angular_speed', 4.0),
                ('image_width', 640),
                ('image_height', 480),
                ('model_input_size', 256),
                ('danger_threshold', 0.65),
                ('critical_threshold', 0.85),
                ('base_speed', 0.5),
                ('use_gpu', True),
                ('debug_publish', True),
                ('use_sim_time', True),
            ]
            for param_name, default_val in params_to_declare:
                if not self.has_parameter(param_name):
                    self.declare_parameter(param_name, default_val)

            # Build config object
            self.config = RLConfig(
                model_path=self.get_parameter('model_path').get_parameter_value().string_value,
                action_type=self.get_parameter('action_type').get_parameter_value().string_value,
                inference_rate=self.get_parameter('inference_rate').get_parameter_value().double_value,
                max_linear_speed=self.get_parameter('max_linear_speed').get_parameter_value().double_value,
                max_angular_speed=self.get_parameter('max_angular_speed').get_parameter_value().double_value,
                image_width=self.get_parameter('image_width').get_parameter_value().integer_value,
                image_height=self.get_parameter('image_height').get_parameter_value().integer_value,
                model_input_size=self.get_parameter('model_input_size').get_parameter_value().integer_value,
                danger_threshold=self.get_parameter('danger_threshold').get_parameter_value().double_value,
                critical_threshold=self.get_parameter('critical_threshold').get_parameter_value().double_value,
                base_speed=self.get_parameter('base_speed').get_parameter_value().double_value,
                use_gpu=self.get_parameter('use_gpu').get_parameter_value().bool_value,
                debug_publish=self.get_parameter('debug_publish').get_parameter_value().bool_value,
            )

            camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
            cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value

            self.get_logger().info("=" * 60)
            self.get_logger().info("  ML & RL Vision Navigation Node Initializing")
            self.get_logger().info("=" * 60)
            self.get_logger().info(f"  Model path: {self.config.model_path or 'AUTO-DETECT'}")
            self.get_logger().info(f"  Inference rate: {self.config.inference_rate} Hz")
            self.get_logger().info(f"  Camera topic: {camera_topic}")
            self.get_logger().info(f"  Cmd vel topic: {cmd_vel_topic}")
            self.get_logger().info("=" * 60)

            # CvBridge
            self.bridge = CvBridge()

            # Thread-safe frame storage
            self._latest_image: Optional[np.ndarray] = None
            self._image_lock = threading.Lock()
            self._dropped_frames = 0
            self._processed_frames = 0

            # QoS configuration
            camera_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )
            cmd_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )

            # Subscribers & Publishers
            self.image_sub = self.create_subscription(Image, camera_topic, self._image_callback, camera_qos)
            self.cmd_vel_pub = self.create_publisher(Twist, cmd_vel_topic, cmd_qos)
            self.debug_pub = self.create_publisher(Float32MultiArray, '/rl_prediction', 10)
            self.fps_pub = self.create_publisher(String, '/rl_diagnostics', 10)

            # Publish initial velocity command to advertise topic immediately
            self.cmd_vel_pub.publish(Twist())

            # Initialize model wrapper
            self.get_logger().info("Initializing ML Model Wrapper...")
            self.model_wrapper = RLModelWrapper(self.config, logger=self.get_logger())
            self.get_logger().info("ML Model Wrapper initialized successfully!")

            # Inference loop timer
            inference_period = 1.0 / self.config.inference_rate
            self.inference_timer = self.create_timer(inference_period, self._inference_loop)

            # Diagnostics timer (1 Hz)
            self.diag_timer = self.create_timer(1.0, self._publish_diagnostics)

            self.get_logger().info("✅ Vision ML Navigation Node ready and waiting for camera feeds...")

        def _image_callback(self, msg: Image):
            """Callback to store incoming ROS image."""
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                with self._image_lock:
                    self._latest_image = cv_image
            except Exception as e:
                self._dropped_frames += 1
                self.get_logger().warning(f"Image conversion failed: {e}")

        def _inference_loop(self):
            """Inference execution loop running at inference_rate Hz."""
            with self._image_lock:
                if self._latest_image is None:
                    self.get_logger().warning("Waiting for camera feed from Gazebo...", throttle_duration_sec=5.0)
                    # Maintain active velocity publisher
                    self.cmd_vel_pub.publish(Twist())
                    return
                cv_image = self._latest_image.copy()

            start_time = time.perf_counter()

            try:
                # 1. Preprocess
                preprocessed = self.model_wrapper.preprocess(cv_image)

                # 2. Predict / Infer depth decision
                decision = self.model_wrapper.predict(preprocessed)

                # 3. Convert decision to Twist
                twist = self.model_wrapper.action_to_twist(decision)

                # 4. Publish velocity command
                self.cmd_vel_pub.publish(twist)

                # 5. Debug info publish
                if self.config.debug_publish:
                    debug_msg = Float32MultiArray()
                    closest = decision.get("closest", 0.0) if isinstance(decision, dict) else 0.0
                    angle = decision.get("steering_angle", 0.0) if isinstance(decision, dict) else 0.0
                    debug_msg.data = [float(twist.linear.x), float(twist.angular.z), float(closest), float(angle)]
                    self.debug_pub.publish(debug_msg)

                self._processed_frames += 1
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                if isinstance(decision, dict):
                    self.get_logger().debug(
                        f"Infer: {elapsed_ms:.1f}ms | Status: {decision.get('status')} | "
                        f"Speed: {twist.linear.x:.2f} | Angular: {twist.angular.z:.2f}",
                        throttle_duration_sec=2.0
                    )

            except Exception as e:
                self.get_logger().error(f"Inference pipeline error: {e}")
                self.cmd_vel_pub.publish(Twist())

        def _publish_diagnostics(self):
            """Publish diagnostics information at 1 Hz."""
            msg = String()
            msg.data = (
                f"fps={self._processed_frames} "
                f"dropped={self._dropped_frames} "
                f"rate={self.config.inference_rate}Hz "
                f"heuristic={self.model_wrapper.is_heuristic}"
            )
            self.fps_pub.publish(msg)
            self._processed_frames = 0

        def destroy_node(self):
            """Safely stop robot on node shutdown."""
            self.get_logger().info("Stopping robot velocity on node shutdown...")
            self.cmd_vel_pub.publish(Twist())
            super().destroy_node()


def main(args=None):
    if not ROS_AVAILABLE:
        print("ROS2 packages not detected in environment. Please source ROS2 setup.bash.")
        sys.exit(1)
    rclpy.init(args=args)
    node = RLNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
