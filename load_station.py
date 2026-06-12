"""Load and simulate the station scene in MuJoCo with interactive viewer."""

import numpy as np
from pathlib import Path
import mujoco
import mujoco.viewer

SCENE_PATH = Path(__file__).parent / "station" / "station.xml"

# Actuator index map (matches <actuator> order in station.xml)
ACTUATOR_NAMES = [
    "left_joint1", "left_joint2",  "left_joint3",
    "left_joint4",  "left_joint5",  "left_joint6",  "left_gripper",
    "right_joint1", "right_joint2", "right_joint3",
    "right_joint4", "right_joint5", "right_joint6", "right_gripper",
]

# Home pose: elbows bent, grippers open
HOME_CTRL = np.array([
    0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,   # left arm
    0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,   # right arm
])


def print_model_info(model: mujoco.MjModel):
    print(f"\n{'='*55}")
    print(f"  Station MuJoCo Model")
    print(f"{'='*55}")
    print(f"  Bodies:    {model.nbody}")
    print(f"  Joints:    {model.njnt}")
    print(f"  Actuators: {model.nu}")
    print(f"  Meshes:    {model.nmesh}")
    print(f"\n  Joint list:")
    for i in range(model.njnt):
        j = model.joint(i)
        lo, hi = model.jnt_range[i]
        jtype = ["free", "ball", "slide", "hinge"][model.jnt_type[i]]
        print(f"    [{i:2d}] {j.name:<20s} {jtype:<6s}  [{lo:7.3f}, {hi:7.3f}]")
    print(f"\n  Actuators:")
    for i in range(model.nu):
        a = model.actuator(i)
        lo, hi = model.actuator_ctrlrange[i]
        print(f"    [{i:2d}] {a.name:<12s}  ctrl=[{lo:.3f}, {hi:.3f}]")
    print(f"{'='*55}\n")


def main():
    print(f"Loading: {SCENE_PATH}")
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data  = mujoco.MjData(model)

    print_model_info(model)

    # Apply home pose
    mujoco.mj_resetData(model, data)
    data.ctrl[:] = HOME_CTRL
    # Step a few times so position controllers settle
    for _ in range(500):
        mujoco.mj_step(model, data)

    print("Launching viewer  (close window to exit)")
    print("  data.ctrl[:7]  → left arm joints + grip")
    print("  data.ctrl[7:]  → right arm joints + grip")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat = [0.5, 0.0, 0.85]
            viewer.cam.distance = 2.0
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 180

            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
    except RuntimeError as e:
        if "mjpython" in str(e):
            print("\n[macOS] launch_passive requires mjpython. Run with:")
            print(f"  mjpython {__file__}")
            print("\nFalling back to blocking launch (no programmatic control)...")
            mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
