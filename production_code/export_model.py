"""
Run this to download the MiDaS_small model and save it locally (state_dict and full model).
"""
import torch
import sys
import os
from pathlib import Path

# Add trusted repos programmatically so torch.hub doesn't prompt in non-interactive environments
try:
    from torch.hub import _TRUSTED_FILTER
    if "rwightman/gen-efficientnet-pytorch" not in _TRUSTED_FILTER:
        _TRUSTED_FILTER.append("rwightman/gen-efficientnet-pytorch")
    if "intel-isl/MiDaS" not in _TRUSTED_FILTER:
        _TRUSTED_FILTER.append("intel-isl/MiDaS")
except Exception:
    pass

PROD_DIR = Path(__file__).parent / "models"
PROD_DIR.mkdir(exist_ok=True)

ROS_DIR = Path(__file__).parent.parent / "src" / "robot_controller" / "models"
ROS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Downloading MiDaS_small (requires internet access)...")
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True, skip_validation=True)
    model.eval()

    print("Saving model weights (state_dict) and full model...")
    # Save state dict
    torch.save(model.state_dict(), str(PROD_DIR / "midas_small_weights.pt"))
    torch.save(model.state_dict(), str(ROS_DIR / "midas_small_weights.pt"))

    # Save full model object
    torch.save(model, str(PROD_DIR / "midas_small.pt"))
    torch.save(model, str(ROS_DIR / "midas_small.pt"))

    print(f"Saved models to production: {PROD_DIR}")
    print(f"Saved models to ROS package: {ROS_DIR}")


if __name__ == "__main__":
    main()