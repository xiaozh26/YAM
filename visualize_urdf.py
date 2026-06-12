"""Load and visualize the station URDF in viser with interactive joint sliders."""

import time
import tempfile
import os
from pathlib import Path

import numpy as np
import yourdfpy
import viser
import viser.extras

URDF_DIR = Path(__file__).parent / "station"
URDF_PATH = URDF_DIR / "station.urdf"


def load_urdf() -> yourdfpy.URDF:
    # Resolve relative mesh paths to absolute so yourdfpy can find them
    content = URDF_PATH.read_text()
    content = content.replace('filename="assets/', f'filename="{URDF_DIR}/assets/')
    with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
        f.write(content)
        tmp_path = f.name
    try:
        urdf = yourdfpy.URDF.load(tmp_path)
    finally:
        os.unlink(tmp_path)
    return urdf


def get_joint_limits(urdf: yourdfpy.URDF) -> dict[str, tuple[float, float]]:
    limits = {}
    for joint in urdf.robot.joints:
        if joint.type == "fixed":
            continue
        lo = joint.limit.lower if joint.limit and joint.limit.lower is not None else -np.pi
        hi = joint.limit.upper if joint.limit and joint.limit.upper is not None else np.pi
        limits[joint.name] = (lo, hi)
    return limits


def main():
    urdf = load_urdf()
    joint_limits = get_joint_limits(urdf)
    joint_names = [j.name for j in urdf.robot.joints if j.type != "fixed"]

    server = viser.ViserServer()
    server.scene.world_axes.visible = True

    viser_urdf = viser.extras.ViserUrdf(server, urdf, root_node_name="/station")

    cfg = np.zeros(len(joint_names))
    viser_urdf.update_cfg(cfg)

    gui_sliders: dict[str, viser.GuiSliderHandle] = {}

    # Partition joints into left arm, right arm, and other
    left_joints = {n: v for n, v in joint_limits.items() if n.startswith("left_")}
    right_joints = {n: v for n, v in joint_limits.items() if n.startswith("right_")}
    other_joints = {
        n: v for n, v in joint_limits.items()
        if not n.startswith("left_") and not n.startswith("right_")
    }

    def add_sliders(folder_name: str, joints: dict[str, tuple[float, float]]):
        with server.gui.add_folder(folder_name):
            for name, (lo, hi) in joints.items():
                initial = max(lo, min(0.0, hi))
                slider = server.gui.add_slider(
                    label=name,
                    min=lo,
                    max=hi,
                    step=0.001,
                    initial_value=initial,
                )
                gui_sliders[name] = slider

    if left_joints:
        add_sliders("Left Arm", left_joints)
    if right_joints:
        add_sliders("Right Arm", right_joints)
    if other_joints:
        add_sliders("Other Joints", other_joints)

    def update_robot(_=None):
        cfg = np.zeros(len(joint_names))
        for i, name in enumerate(joint_names):
            if name in gui_sliders:
                cfg[i] = gui_sliders[name].value
        viser_urdf.update_cfg(cfg)

    for slider in gui_sliders.values():
        slider.on_update(update_robot)

    print(f"Viser running at http://localhost:{server.get_port()}")
    print(f"Loaded {len(joint_names)} movable joints: {len(left_joints)} left, {len(right_joints)} right, {len(other_joints)} other")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
