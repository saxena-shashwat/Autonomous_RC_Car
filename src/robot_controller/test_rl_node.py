"""
Unit and integration test script for RL/ML Vision controller node wrapper.
"""
import sys
import os
import numpy as np
import cv2

# Set stdout encoding if needed
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add robot_controller module path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "robot_controller"))

from rl_node import RLConfig, RLModelWrapper, Navigator, SlidingWindowCompressor


def test_compressor():
    print("[TEST 1] Testing SlidingWindowCompressor...")
    compressor = SlidingWindowCompressor(window_size=3)
    f1 = np.full((10, 10, 3), 100, dtype=np.uint8)
    f2 = np.full((10, 10, 3), 120, dtype=np.uint8)
    f3 = np.full((10, 10, 3), 140, dtype=np.uint8)

    _ = compressor.add_and_get(f1)
    _ = compressor.add_and_get(f2)
    avg = compressor.add_and_get(f3)

    expected = 120
    assert np.allclose(avg, expected), f"Expected {expected}, got {avg[0,0,0]}"
    print("  [OK] SlidingWindowCompressor test passed!")


def test_navigator():
    print("[TEST 2] Testing Navigator decision engine...")
    config = RLConfig(
        danger_threshold=0.65,
        critical_threshold=0.85,
        base_speed=0.5,
        min_speed=0.1,
        max_steering_angle=60.0
    )
    nav = Navigator(config)

    # 1. Clear depth map (low closeness)
    clear_depth = np.full((100, 300), 0.2, dtype=np.float32)
    d_clear = nav.decide(clear_depth)
    assert d_clear["status"] == "CLEAR", f"Expected CLEAR, got {d_clear['status']}"
    assert np.isclose(d_clear["speed"], 0.5), f"Expected speed 0.5, got {d_clear['speed']}"
    assert np.isclose(d_clear["steering_angle"], 0.0), f"Expected angle 0, got {d_clear['steering_angle']}"

    # 2. Critical obstacle (very close depth > 0.85)
    crit_depth = np.full((100, 300), 0.9, dtype=np.float32)
    d_crit = nav.decide(crit_depth)
    assert d_crit["status"] == "EMERGENCY_STOP", f"Expected EMERGENCY_STOP, got {d_crit['status']}"
    assert d_crit["speed"] == 0.0, f"Expected speed 0.0, got {d_crit['speed']}"

    # 3. Left obstacle (left zone high closeness, right zone low)
    left_obs_depth = np.full((100, 300), 0.2, dtype=np.float32)
    left_obs_depth[30:90, 0:100] = 0.75  # left zone obstacle
    d_left = nav.decide(left_obs_depth)
    assert d_left["status"] == "TURN_RIGHT", f"Expected TURN_RIGHT, got {d_left['status']}"
    assert d_left["steering_angle"] > 0, f"Expected positive steering angle (steer right), got {d_left['steering_angle']}"

    # 4. Right obstacle (right zone high closeness, left zone low)
    right_obs_depth = np.full((100, 300), 0.2, dtype=np.float32)
    right_obs_depth[30:90, 200:300] = 0.75  # right zone obstacle
    d_right = nav.decide(right_obs_depth)
    assert d_right["status"] == "TURN_LEFT", f"Expected TURN_LEFT, got {d_right['status']}"
    assert d_right["steering_angle"] < 0, f"Expected negative steering angle (steer left), got {d_right['steering_angle']}"

    print("  [OK] Navigator decision engine tests passed!")


def test_model_wrapper():
    print("[TEST 3] Testing RLModelWrapper pipeline...")
    config = RLConfig(
        image_width=640,
        image_height=480,
        base_speed=0.5,
        max_linear_speed=2.0,
        max_angular_speed=4.0
    )
    wrapper = RLModelWrapper(config)

    # Test image frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_frame, (50, 100), (200, 400), (255, 255, 255), -1)

    preprocessed = wrapper.preprocess(test_frame)
    decision = wrapper.predict(preprocessed)
    twist = wrapper.action_to_twist(decision)

    assert hasattr(twist, 'linear') and hasattr(twist, 'angular'), "Output is not valid Twist message"
    print(f"  Inference Result -> Status: {decision.get('status')} | Linear.x: {twist.linear.x:.2f} m/s | Angular.z: {twist.angular.z:.2f} rad/s")
    print("  [OK] RLModelWrapper pipeline test passed!")


if __name__ == "__main__":
    print("==================================================")
    print(" Running Vision ML Model & Controller Node Tests")
    print("==================================================")
    test_compressor()
    test_navigator()
    test_model_wrapper()
    print("==================================================")
    print(" ALL TESTS PASSED SUCCESSFULLY!")
    print("==================================================")
