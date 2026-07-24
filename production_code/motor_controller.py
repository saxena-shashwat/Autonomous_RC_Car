"""
Hardware abstraction layer for steering + drive motors.

YOU WILL NEED TO EDIT THIS FILE to match your actual wiring - pin numbers,
whether you're using servo-based steering (RC car style) or differential
drive (tank steering with two motors), and which driver IC you have
(L298N, TB6612FNG, etc). What's here is a working default for the common
case of: one steering servo + one drive motor through a simple H-bridge.

If gpiozero isn't installed or no hardware is detected (e.g. you're
testing on a laptop), this falls back to a "simulation" mode that just
logs the commands it would have sent - so you can test the vision/decision
logic before ever touching real hardware.
"""
from config import get_logger

logger = get_logger("nav.motor")

try:
    from gpiozero import Servo, Motor
    GPIO_AVAILABLE = True
except Exception:
    GPIO_AVAILABLE = False
    logger.warning("gpiozero not available -> running in SIMULATION mode (no real hardware control).")


class MotorController:
    def __init__(self, steering_pin=18, motor_fwd_pin=23, motor_bwd_pin=24, enable_pin=25):
        self.simulated = not GPIO_AVAILABLE

        if not self.simulated:
            try:
                # min/max_pulse_width are servo-specific - check your servo's datasheet
                # and adjust if full-left/full-right don't match your physical linkage.
                self.steering = Servo(
                    steering_pin, min_pulse_width=0.5 / 1000, max_pulse_width=2.5 / 1000
                )
                self.drive = Motor(
                    forward=motor_fwd_pin, backward=motor_bwd_pin, enable=enable_pin, pwm=True
                )
                logger.info("Motor hardware initialized (gpiozero).")
            except Exception as e:
                logger.error(f"Failed to init GPIO hardware ({e}); falling back to simulation mode.")
                self.simulated = True

    def set_steering(self, angle_deg: float):
        """angle_deg expected in [-60, 60]. Negative = left, positive = right."""
        angle_deg = max(-60.0, min(60.0, angle_deg))
        servo_value = angle_deg / 60.0  # gpiozero Servo wants a value in [-1, 1]
        if self.simulated:
            logger.debug(f"[SIM] steering -> {angle_deg:.1f} deg")
        else:
            self.steering.value = servo_value

    def set_speed(self, speed: float):
        """speed expected in [0.0, 1.0]. This simple rover only drives forward;
        add a reverse/backward path here if your project needs it."""
        speed = max(0.0, min(1.0, speed))
        if self.simulated:
            logger.debug(f"[SIM] speed -> {speed:.2f}")
        else:
            if speed <= 0.0:
                self.drive.stop()
            else:
                self.drive.forward(speed)

    def stop(self):
        """Emergency / shutdown stop - always safe to call."""
        self.set_speed(0.0)
        self.set_steering(0.0)
        if not self.simulated:
            self.drive.stop()