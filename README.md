# Vision-Based Autonomous Robot - ROS2 Jazzy Workspace

## Quick Start

### 1. Prerequisites

```bash
# Ubuntu 24.04 LTS with ROS2 Jazzy installed
# Gazebo Harmonic (gz-sim)
# Python dependencies
sudo apt update
sudo apt install ros-jazzy-ros-gz ros-jazzy-cv-bridge python3-opencv python3-torch
```

### 2. Build the Workspace

```bash
cd ~/ros2_ws
colcon build --packages-select robot_description robot_sim robot_controller robot_bringup
source install/setup.bash
```

### 3. Launch Everything (One Command)

```bash
# Full simulation with RL node
ros2 launch robot_bringup simulation.launch.py

# With custom world
ros2 launch robot_bringup simulation.launch.py world_file:=my_world.world

# With your trained model
ros2 launch robot_bringup simulation.launch.py model_path:=/path/to/your/model.pt

# Without RL (manual control)
ros2 launch robot_bringup simulation.launch.py use_rl:=false

# Without RViz
ros2 launch robot_bringup simulation.launch.py use_rviz:=false
```

### 4. Run RL Node Independently

```bash
# If Gazebo is already running and you just want to start the RL node:
ros2 launch robot_controller rl_node.launch.py model_path:=/path/to/model.pt

# Or directly:
ros2 run robot_controller rl_node --ros-args -p model_path:="/path/to/model.pt"
```

### 5. Manual Control (Teleop)

```bash
# If RL is disabled, use keyboard teleop
ros2 run teleop_twist_keyboard teleop_twist_keyboard

# Or publish manually
ros2 topic pub /cmd_vel geometry_msgs/Twist '{linear: {x: 0.5}, angular: {z: 0.0}}'
```

---

## Integrating YOUR RL Model

The only file you need to modify is:

**`robot_controller/robot_controller/rl_node.py`**

Look for these `TODO` sections:

### 1. Model Loading (`_load_model`)

Replace the placeholder with your model loading code:

```python
def _load_model(self):
    import torch
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Option A: Load PyTorch model directly
    self.model = torch.load(self.config.model_path, map_location=self.device)
    self.model.eval()

    # Option B: Stable-Baselines3
    from stable_baselines3 import PPO
    self.model = PPO.load(self.config.model_path, device=self.device)

    # Option C: TensorFlow
    import tensorflow as tf
    self.model = tf.keras.models.load_model(self.config.model_path)
```

### 2. Preprocessing (`preprocess`)

Adapt to your model's expected input:

```python
def preprocess(self, cv_image):
    import torch
    import torchvision.transforms as T

    # Resize
    image = cv2.resize(cv_image, (224, 224))
    # BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # To tensor and normalize
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0).to(self.device)
    return tensor
```

### 3. Inference (`predict`)

Replace with your model's forward pass:

```python
def predict(self, preprocessed_input):
    with torch.no_grad():
        if hasattr(self.model, 'predict'):
            # Stable-Baselines3 style
            action, _ = self.model.predict(preprocessed_input, deterministic=True)
            return action
        else:
            # Raw PyTorch model
            output = self.model(preprocessed_input)
            # Extract action from output
            return output.cpu().numpy()
```

### 4. Action Space (`action_to_twist`)

Already supports:
- **Continuous**: `[linear_vel, angular_vel]` → maps directly to Twist
- **Discrete**: `0=LEFT, 1=RIGHT, 2=FORWARD` → converts to Twist

Set via launch parameter: `action_type:=continuous` or `action_type:=discrete`

---

## Architecture

```
Gazebo Harmonic
    ├── Robot Model (URDF/Xacro)
    ├── Camera Plugin → /camera/image_raw
    └── Diff Drive Plugin ← /cmd_vel

ROS2 Jazzy
    ├── robot_state_publisher (TF tree)
    ├── ros_gz_bridge (Gazebo ↔ ROS topics)
    └── RL Node
         ├── Subscribes /camera/image_raw
         ├── CvBridge → OpenCV → PyTorch
         ├── Model Inference
         └── Publishes /cmd_vel
```

## Topic List

| Topic | Type | Direction | Description |
|-------|------|-----------|-------------|
| `/camera/image_raw` | sensor_msgs/Image | Sub | Camera feed from Gazebo |
| `/camera/camera_info` | sensor_msgs/CameraInfo | Sub | Camera calibration |
| `/cmd_vel` | geometry_msgs/Twist | Pub | Velocity commands to robot |
| `/odom` | nav_msgs/Odometry | Pub | Robot odometry |
| `/rl_prediction` | Float32MultiArray | Pub | Debug: [linear, angular] |
| `/rl_diagnostics` | String | Pub | FPS and frame drop info |

## Parameters

All configurable via launch arguments or ROS2 params:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_path` | `""` | Path to trained RL model file |
| `action_type` | `"continuous"` | `"continuous"` or `"discrete"` |
| `inference_rate` | `20.0` | Hz, how often to run inference |
| `max_linear_speed` | `2.0` | m/s cap on forward/backward |
| `max_angular_speed` | `4.0` | rad/s cap on rotation |
| `camera_topic` | `/camera/image_raw` | Input image topic |
| `cmd_vel_topic` | `/cmd_vel` | Output velocity topic |
| `use_sim_time` | `true` | Use Gazebo simulation clock |

## Debugging

```bash
# Check if camera is publishing
ros2 topic hz /camera/image_raw

# View camera feed
ros2 run rqt_image_view rqt_image_view

# Monitor RL outputs
ros2 topic echo /rl_prediction

# Check TF tree
ros2 run tf2_tools view_frames

# Node info
ros2 node info /rl_node
```
