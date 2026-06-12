# YAM Station — Bimanual MuJoCo Simulation & VR Teleoperation

> A project by the **Embodied Intelligence & Robotics Center (EMBER), UC Berkeley**

**YAM Station** is a bimanual robot arm simulation and VR teleoperation stack built on MuJoCo.
Control two 6-DOF YAM arms — left and right — from a **Meta Quest 3** headset in real time using a full 6-DOF Cartesian differential IK (damped-least-squares Jacobian), giving the gripper precise position *and* orientation tracking that follows your hand.

---

## Repo Structure

```
YAM/
├── load_station.py            # Load station.xml in MuJoCo viewer (home pose demo)
├── visualize_urdf.py          # Visualize station URDF with interactive joint sliders (viser)
└── station/
    ├── station.xml            # MuJoCo bimanual scene (authoritative model)
    ├── station.urdf           # URDF for visualization / planning tooling
    ├── scene.xml              # Alternate scene entry point
    ├── vr_teleop_mujoco.py    # ★ Main VR teleop script (Quest 3 → MuJoCo)
    ├── yam_assets/            # YAM arm STL meshes (visual + collision)
    ├── gripper_assets/        # Gripper STL meshes
    ├── assets/                # Table, camera mount, and other scene meshes
    └── vr/                    # Self-contained VR layer (XLeVR, vendored)
        ├── config.yaml        # Network / robot config
        ├── vr_monitor.py      # WebSocket server that ingests Quest 3 hand data
        ├── xlevr/             # XLeVR Python package (vendored)
        └── web-ui/            # Quest 3 browser VR app (WebXR)
```

---

## Overview

This project adds a VR teleoperation layer on top of a MuJoCo bimanual station.
Instead of scripted trajectories, you stream live hand-tracking data from a Quest 3 to both simulated arms over a Cloudflare tunnel — no local network pairing required.

**Key design choices:**

| Feature | Detail |
|---|---|
| IK solver | Damped-least-squares Jacobian (6-DOF position + orientation) |
| Control mode | GRIP clutch: arm moves only while grip is held — reposition freely |
| Orientation tracking | Full 6-DOF: gripper rotation mirrors the controller's rotation in world frame |
| Axis remap | VR frame → robot world frame via a fixed 3×3 basis matrix `B_VR_TO_WORLD` |
| Gripper | Trigger button → close; release → open (ctrl: 0.0 = closed, 0.041 = open) |
| Double-click grip | Arms smoothly ramp back to the rest pose (cosine ease-in-out, 1.5 s) |
| Latency path | Quest 3 browser → Cloudflare tunnel → local aiohttp server → MuJoCo control loop |
| Supported configs | Single arm (`left`/`right`) · Both arms simultaneous |
| Physics | MuJoCo passive viewer, 60 Hz control, 8 substeps per tick |

---

## Prerequisites

### 1. Python environment

```bash
python -m venv .venv && source .venv/bin/activate   # or use conda
```

### 2. Install Python dependencies

```bash
pip install mujoco aiohttp viser yourdfpy numpy
```

### 3. Install Cloudflared

Cloudflared creates a public HTTPS tunnel so the Quest 3 browser can reach your local machine without network configuration.

```bash
brew install cloudflared      # macOS
# Linux / other: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
```

### 4. (macOS only) Install `mjpython`

MuJoCo's passive viewer on macOS requires the bundled `mjpython` launcher:

```bash
pip install mujoco
# mjpython is installed alongside mujoco — verify with:
mjpython --version
```

---

## Usage

### Load the station (no headset required)

Launches the MuJoCo interactive viewer with both arms in the home pose.

```bash
mjpython load_station.py          # macOS (requires mjpython)
python  load_station.py           # Linux
```

Prints a table of all joints, actuators, and their control ranges on startup.

### Visualize the URDF with joint sliders

Opens a `viser` web UI at `http://localhost:<port>` with interactive sliders for every
movable joint, partitioned into Left Arm / Right Arm / Other folders.

```bash
python visualize_urdf.py
```

### VR teleoperation (main script)

```bash
# Both arms (default)
mjpython station/vr_teleop_mujoco.py

# Single arm
mjpython station/vr_teleop_mujoco.py --arm right
mjpython station/vr_teleop_mujoco.py --arm left

# No headset — drive the EE in a circle to validate IK + viewer
mjpython station/vr_teleop_mujoco.py --test
```

Once running you will see:

```
🌐 VR server on port 8080
┌──────────────────────────────────────┐
│  Quest 3 browser → type this:        │
│  http://192.168.x.x:8080/go          │
│  It redirects to the VR page; then   │
│  tap the VR goggles icon & squeeze   │
│  grip to start controlling.          │
└──────────────────────────────────────┘
```

### All CLI flags (`vr_teleop_mujoco.py`)

| Flag | Default | Description |
|---|---|---|
| `--arm` | `both` | `left` · `right` · `both` |
| `--control_hz` | `60.0` | Control loop frequency (Hz) |
| `--pos-scale` | `1.0` | Hand metres → robot metres (`< 1` = finer motion) |
| `--no-orientation` | — | Position-only tracking; gripper keeps its grip-time orientation |
| `--home_seconds` | `1.5` | Duration of the double-click-grip homing ramp |
| `--test` | — | Headless IK self-test: drives EE in a circle, no Quest needed |

---

## Connecting the Quest 3

1. Start `vr_teleop_mujoco.py` and wait for the URL box to appear.
2. On the Quest 3, open the **Meta browser**.
3. Type the local URL shown (e.g. `http://192.168.x.x:8080/go`).
   It redirects through the Cloudflare tunnel to the VR web UI.
4. Tap the **VR goggles icon** to enter immersive mode.
5. Squeeze a controller grip button to start controlling.

The arms will **not move** until you squeeze grip — the first grip anchors the origin at the current rest pose and holds.

---

## Controls

| Action | How |
|---|---|
| **Move arm** | Squeeze **GRIP**, then move your hand |
| **Freeze / reposition hand** | Release grip — arm stops and holds; reposition freely, then grip again |
| **Gripper close** | Pull **TRIGGER** |
| **Gripper open** | Release trigger |
| **Return to home pose** | Double-click grip (two quick presses within 0.4 s) |

> The arm **only moves while you hold grip** (grip clutch mechanic).
> Releasing grip freezes the arm so you can reposition your hand without disturbing the robot —
> exactly like lifting a mouse off a pad.

---

## How It Works

```
Quest 3 (WebXR, ~72 Hz)
    │  controller position + quaternion + grip + trigger
    ▼
Cloudflare Tunnel  ──►  aiohttp server (port 8080)
                              │  WebSocket proxy
                              ▼
                         VRMonitor (XLeVR, vendored)
                              │  get_latest_goal_nowait()
                              ▼
                    VRArmMapper (per arm, 60 Hz)
                      ├─ GRIP rising edge → anchor VR origin + EE pose
                      ├─ hand disp × B_VR_TO_WORLD → Δposition in robot frame
                      ├─ relative quaternion × B_VR_TO_WORLD → Δorientation
                      └─ target_pos + target_R
                              │
                         ArmIK (damped least-squares)
                           J = [Jp; Jr]  (6×6 per arm)
                           dq = Jᵀ(JJᵀ + λ²I)⁻¹ err   (8 iters)
                              │
                    MuJoCo control loop (8 substeps / tick)
                              │
                         Passive viewer (sync)
```

**Grip clutch**: only the *displacement* from the grip-capture origin drives the arm,
so connecting mid-motion or repositioning your hand never causes a jump.

**Orientation tracking**: the relative quaternion from the grip-capture frame is remapped
from the WebXR controller frame into the robot world frame via a similarity transform
`R_world = B · R_vr · Bᵀ`, then composed onto the base EE orientation captured at grip time.

**Damped-least-squares IK**:
```
dq = Jᵀ (J Jᵀ + λ²I)⁻¹ err,   λ = 0.12
```
Joint limits are clamped each iteration. The step size is capped at 0.25 rad/joint/tick.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `launch_passive requires mjpython` | Run with `mjpython` instead of `python` on macOS |
| `Could not import vr_monitor` | Ensure `station/vr/` is intact (vendored inside this repo) |
| `cloudflared` not found | `brew install cloudflared` |
| Arm jumps on first VR frame | Already handled — first grip only sets the anchor, no motion |
| Gripper drifts closed | Check `GRIP_OPEN_CTRL = 0.041` matches your `station.xml` actuator range |
| Quest browser shows blank page | Wait ~5 s for the Cloudflare tunnel to initialize |
| IK doesn't converge | Increase `--ik_iters` or reduce `--pos-scale` to keep targets reachable |
| Arm moves inverted on one axis | Flip the corresponding element in `VRTeleopConfig.sign` |

---

## Acknowledgements

Built on top of:
- [MuJoCo](https://github.com/google-deepmind/mujoco) by Google DeepMind
- [XLeRobot / XLeVR](https://github.com/Vector-Wangel/XLeRobot) by Vector-Wangel (VR layer vendored under `station/vr/`)
- [LeRobot](https://github.com/huggingface/lerobot) by Hugging Face (structural inspiration)
- [viser](https://github.com/nerfstudio-project/viser) for interactive 3D visualization

---

## License

[MIT](LICENSE) © 2026 Embodied Intelligence & Robotics Center (EMBER), UC Berkeley
