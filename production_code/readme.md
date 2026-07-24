# Rover Depth Navigation

Continuous camera -> MiDaS depth -> obstacle-avoidance steering for a
Raspberry Pi rover.

## Files
- `config.py` - all tunable parameters in one place
- `export_model.py` - **run once, with internet**, to cache the model offline
- `navigation_car.py` - the main loop you run on the rover
- `motor_controller.py` - hardware interface; **edit this for your wiring**
- `requirements.txt`

## Setup

1. On a laptop/desktop with internet access:
   ```
   pip install torch
   python export_model.py
   ```
   This downloads MiDaS_small once and saves `models/midas_small_ts.pt` -
   a self-contained file. The Pi never needs internet or `torch.hub` again.

2. Copy the entire `rover_nav/` folder (including `models/`) onto the Pi.

3. On the Pi:
   ```
   pip install -r requirements.txt
   ```
   Note: plain `pip install torch` on Raspberry Pi OS can be slow/unreliable
   depending on your OS version and Pi model. If it fails, look at
   [piwheels](https://www.piwheels.org/) for a prebuilt ARM wheel matching
   your Python version, or consider the ONNX Runtime alternative below.

4. Open `motor_controller.py` and set the GPIO pin numbers to match your
   actual wiring (steering servo pin, drive motor forward/backward/enable
   pins). The defaults assume one steering servo + one H-bridge-driven motor.

5. Run it:
   ```
   python navigation_car.py            # headless - for the actual rover
   python navigation_car.py --show     # with debug windows - needs a monitor
   ```

   Press Ctrl+C to stop; motors are cut cleanly on shutdown.

## What changed vs. your original scripts

- **Continuous operation**: replaced the "load one image, wait for keypress"
  flow with a threaded camera reader + a real-time loop that runs until
  interrupted.
- **Offline model loading**: `torch.hub.load()` was being called at every
  startup, which hits GitHub and is slow. `export_model.py` now traces the
  model once into a TorchScript file that loads instantly, offline.
- **`base_speed` was computed but never used** in either of your scripts.
  It's now wired into an actual speed decision that slows the rover down
  the closer/sharper an obstacle gets, instead of always driving at a fixed
  speed.
- **No emergency stop**: neither version had a "something is directly in
  front of me" case beyond steering away from it. Added a
  `CRITICAL_THRESHOLD` that cuts speed to zero if an obstacle is very close,
  regardless of steering direction.
- **Blocking `cv2.waitKey(0)`**: fine for a one-shot demo, fatal for a
  continuous loop. Debug display is now optional (`--show`) and non-blocking.
- **No fail-safe on inference/camera errors**: added try/except around
  inference and camera reads so a dropped frame or a model hiccup stops the
  rover instead of driving on stale/garbage data.
- **Merged the "dynamic turning" logic** (90th-percentile zone closeness +
  closeness-squared sensitivity, cropped focus band instead of just the
  bottom half) from your second script, since it reacts more smoothly than
  the plain-mean version.

## Honest limitations to test before trusting this on anything that moves

- All thresholds (`DANGER_THRESHOLD`, `CRITICAL_THRESHOLD`, `SENSITIVITY`,
  speeds) are starting points, not calibrated values - MiDaS's relative
  depth scale changes with scene content, so test extensively in your actual
  environment before increasing speed.
- MiDaS_small on a Raspberry Pi CPU is not fast. `TARGET_FPS = 10` is
  optimistic on a Pi 4; measure your actual inference time and adjust, or
  look into exporting to ONNX Runtime (`onnxruntime`) instead of full
  PyTorch - it's typically 2-4x faster on ARM CPUs for a model this size.
- A single monocular depth model has no true scale or velocity information -
  it can't tell you *how fast* something is approaching, only how close it
  looks in the current frame. For anything beyond a slow indoor hobby rover,
  pair this with an ultrasonic/ToF/LIDAR sensor as a hard safety backstop
  that can force-stop the motors independent of the vision pipeline.