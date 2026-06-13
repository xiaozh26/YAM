"""
VR Teleoperation → YAM Station MuJoCo Simulation
================================================
Drive the bimanual YAM station (station/station.xml) in a MuJoCo viewer with a
Meta Quest 3. Move a controller and the simulated arm's gripper follows the
controller's position + rotation in real time.

This is the MuJoCo/YAM analog of lerobot/scripts/vr_teleop_sim.py. It keeps the
SAME VR input layer (XLeVR `VRMonitor` websocket server + cloudflared tunnel) and
the SAME control UX (GRIP = clutch, TRIGGER = gripper), but the "control brain"
is different on purpose:

    SO-101  (LeRobot sim) : 5-DOF underactuated arm → 2-link planar IK + wrist hacks.
    YAM     (this script) : 6-DOF arm  → proper Cartesian differential IK so the
                            gripper tracks the controller's full 6-DOF pose
                            (position AND orientation) via the MuJoCo Jacobian
                            (damped least-squares). This is strictly more capable
                            than the SO-101 planar hack, so we use it.

Control = GRIP CLUTCH (identical feel to the LeRobot sim):
  • The arm holds its current pose until you SQUEEZE GRIP.
  • Grip press anchors the origin at the current pose WITHOUT moving (so the first
    grip starts exactly from the loaded rest pose); while held, your hand moves the
    gripper relative to that anchor.
  • Releasing grip FREEZES the arm so you can reposition your hand and grip again
    (like lifting a mouse).
  • DOUBLE-CLICK GRIP (two quick grip presses) → the arm(s) smoothly return to the
    home pose and teleop starts over — same gesture as xlerobot's vr_teleop_real.py.
  • TRIGGER closes/opens the gripper.

Usage:
    python station/vr_teleop_mujoco.py                 # both arms
    python station/vr_teleop_mujoco.py --arm right     # single arm
    python station/vr_teleop_mujoco.py --test          # no headset: IK self-test

  1. Watch the sim on your computer: the MuJoCo passive viewer window opens.
  2. A cloudflared tunnel for the Quest is started automatically. In the Quest 3
     browser, type the short  http://<your-ip>:8080/go  URL it prints — it
     redirects to the VR page. Tap the VR goggles icon, then squeeze grip.
     (If cloudflared isn't installed:  brew install cloudflared )

Motion mapping (fixed WORLD frame): push the controller forward → the gripper
reaches forward (robot +x); move right/left/up/down → gripper right/left/up/down;
rotate the controller → the gripper rotates the same way. Sign knobs below if any
axis looks mirrored.
"""

import argparse
import asyncio
import math
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import aiohttp
import mujoco
import mujoco.viewer
import numpy as np
from aiohttp import web

# ── Paths ────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "station.xml")

# Self-contained VR layer vendored under station/vr/ (xlevr pkg + vr_monitor +
# web-ui). Nothing here depends on an external XLeVR checkout.
VR_DIR = os.path.join(_HERE, "vr")
WEB_UI_DIR = os.path.join(VR_DIR, "web-ui")

COMBINED_PORT = 8080   # VR web page + WS proxy (this is the cloudflared target)

if VR_DIR not in sys.path:
    sys.path.insert(0, VR_DIR)

try:
    from vr_monitor import VRMonitor, get_local_ip
except ImportError as e:
    print(f"[ERROR] Could not import the vendored vr_monitor from {VR_DIR}: {e}")
    print("        Ensure station/vr/ (xlevr + vr_monitor.py) is intact.")
    VRMonitor = None
    def get_local_ip():  # type: ignore
        return "localhost"


# ─────────────────────────────────────────────────────────────────────────────
# VR ↔ robot-world axis remap + control knobs
# ─────────────────────────────────────────────────────────────────────────────
# WebXR controller frame:  +x = right, +y = up, +z = toward the user (back).
# YAM station world frame:  +x = forward (toward table), +y = left, +z = up.
# B maps a vector expressed in the VR frame into the robot-world frame so that a
# physical hand motion drives the matching gripper motion. Columns = images of the
# VR basis vectors in the robot frame:
#     VR +x (right) → robot (0,-1, 0)   (right = −y)
#     VR +y (up)    → robot (0, 0, 1)   (up    = +z)
#     VR +z (back)  → robot (-1,0, 0)   (back  = −x  ⇒ forward push = +x reach)
B_VR_TO_WORLD = np.array([
    [0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])

# Gripper actuator ctrl: 0.0 = closed, 0.041 = open (measured on station.xml).
GRIP_OPEN_CTRL = 0.041
GRIP_CLOSED_CTRL = 0.0

# Home / rest pose (6 arm-joint radians per arm). All-zeros = the model's loaded
# rest pose (gripper points forward over the table). DOUBLE-CLICK GRIP ramps the
# arm back here, mirroring xlerobot's grip double-click "return to home".
HOME_QPOS = np.zeros(6)

# Two grip presses within this window (s) → return to home (same as xlerobot).
DOUBLE_CLICK_WINDOW = 0.4

# Per-arm MuJoCo names in station.xml.
ARM_SPEC = {
    "left": {
        "joints": [f"left_joint{i}" for i in range(1, 7)],
        "site": "left_tcp_site",
        "arm_acts": [f"left_joint{i}" for i in range(1, 7)],
        "grip_act": "left_gripper",
    },
    "right": {
        "joints": [f"right_joint{i}" for i in range(1, 7)],
        "site": "right_tcp_site",
        "arm_acts": [f"right_joint{i}" for i in range(1, 7)],
        "grip_act": "right_gripper",
    },
}


@dataclass
class VRTeleopConfig:
    arm: str = "both"            # "left" | "right" | "both"
    control_hz: float = 60.0
    physics_substeps: int = 8    # mj_step per control tick (timestep 0.002 → ~0.96ms*8)
    gripper_trigger_threshold: float = 0.5
    home_seconds: float = 1.5        # duration of the smooth double-click homing ramp
    track_orientation: bool = True   # False = position-only (gripper keeps grip orientation)
    pos_scale: float = 1.0       # hand metres → robot metres
    # IK (damped least-squares)
    ik_iters: int = 8
    ik_damping: float = 0.12
    ik_pos_tol: float = 1e-3
    ik_max_step: float = 0.25    # rad per joint per control tick (rate limit)
    # Per-axis sign knobs (flip if a translation axis looks mirrored)
    sign: tuple = (1.0, 1.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Small quaternion / rotation helpers (MuJoCo wxyz convention)
# ─────────────────────────────────────────────────────────────────────────────

def mat_to_quat(mat9: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(mat9, dtype=np.float64).ravel())
    return q


def quat_to_rotvec(quat: np.ndarray) -> np.ndarray:
    """Quaternion (wxyz) → rotation vector (axis * angle), world frame."""
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, np.asarray(quat, dtype=np.float64), 1.0)
    return v


def rotmat_error(R_tgt: np.ndarray, R_cur: np.ndarray) -> np.ndarray:
    """World-frame orientation error (3-vec) that rotates R_cur onto R_tgt."""
    R_err = R_tgt @ R_cur.T
    return quat_to_rotvec(mat_to_quat(R_err))


def quat_from_xyzw(q_xyzw) -> np.ndarray:
    """WebXR/Three.js quaternion (xyzw) → MuJoCo wxyz."""
    x, y, z, w = q_xyzw
    return np.array([w, x, y, z], dtype=np.float64)


def remap_quat_to_world(q_wxyz: np.ndarray) -> np.ndarray:
    """Express a VR-frame rotation as the equivalent robot-world rotation:
    R_world = B · R_vr · Bᵀ  (similarity transform), returned as a wxyz quat."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q_wxyz, dtype=np.float64))
    R = R.reshape(3, 3)
    R_world = B_VR_TO_WORLD @ R @ B_VR_TO_WORLD.T
    return mat_to_quat(R_world)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    r = np.zeros(4)
    mujoco.mju_mulQuat(r, np.asarray(a, np.float64), np.asarray(b, np.float64))
    return r


def quat_conj(a: np.ndarray) -> np.ndarray:
    r = np.zeros(4)
    mujoco.mju_negQuat(r, np.asarray(a, np.float64))
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Per-arm Cartesian differential-IK controller
# ─────────────────────────────────────────────────────────────────────────────

class ArmIK:
    """Holds the index bookkeeping + a scratch MjData for one arm, and turns a
    target EE pose into joint commands via damped-least-squares Jacobian IK."""

    def __init__(self, model: mujoco.MjModel, side: str, cfg: VRTeleopConfig):
        self.model = model
        self.cfg = cfg
        self.side = side
        spec = ARM_SPEC[side]

        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, spec["site"])
        self.dofs = []
        self.qadr = []
        self.jnt_range = []
        for jn in spec["joints"]:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            self.dofs.append(model.jnt_dofadr[jid])
            self.qadr.append(model.jnt_qposadr[jid])
            self.jnt_range.append(tuple(model.jnt_range[jid]))
        self.dofs = np.array(self.dofs)
        self.qadr = np.array(self.qadr)

        self.arm_act = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
                        for a in spec["arm_acts"]]
        self.grip_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                          spec["grip_act"])

        self._scratch = mujoco.MjData(model)
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

        # Commanded joint angles for this arm (seeded at startup from current pose).
        self.q_cmd = np.zeros(6)
        self.grip_ctrl = GRIP_OPEN_CTRL

    # -- forward kinematics of the current commanded pose ---------------------
    def fk(self, full_qpos: np.ndarray):
        d = self._scratch
        d.qpos[:] = full_qpos
        d.qpos[self.qadr] = self.q_cmd
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)
        pos = d.site_xpos[self.site_id].copy()
        R = d.site_xmat[self.site_id].reshape(3, 3).copy()
        return pos, R

    def seed_from(self, full_qpos: np.ndarray):
        self.q_cmd = full_qpos[self.qadr].copy()

    # -- damped-least-squares step toward a target EE pose --------------------
    def solve(self, full_qpos: np.ndarray, target_pos: np.ndarray,
              target_R: Optional[np.ndarray]):
        cfg = self.cfg
        d = self._scratch
        lam2 = cfg.ik_damping ** 2
        q = self.q_cmd.copy()
        use_ori = target_R is not None

        for _ in range(cfg.ik_iters):
            d.qpos[:] = full_qpos
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)

            p_cur = d.site_xpos[self.site_id]
            perr = target_pos - p_cur

            mujoco.mj_jacSite(self.model, d, self._jacp, self._jacr, self.site_id)
            Jp = self._jacp[:, self.dofs]

            if use_ori:
                R_cur = d.site_xmat[self.site_id].reshape(3, 3)
                oerr = rotmat_error(target_R, R_cur)
                J = np.vstack([Jp, self._jacr[:, self.dofs]])
                err = np.concatenate([perr, oerr])
            else:
                J = Jp
                err = perr

            # dq = Jᵀ (J Jᵀ + λ²I)⁻¹ err
            JJt = J @ J.T
            dq = J.T @ np.linalg.solve(JJt + lam2 * np.eye(JJt.shape[0]), err)
            dq = np.clip(dq, -cfg.ik_max_step, cfg.ik_max_step)
            q = q + dq

            # respect joint limits
            for i, (lo, hi) in enumerate(self.jnt_range):
                q[i] = min(max(q[i], lo), hi)

            if np.linalg.norm(perr) < cfg.ik_pos_tol and not use_ori:
                break

        self.q_cmd = q

    def apply(self, data: mujoco.MjData):
        for act, qv in zip(self.arm_act, self.q_cmd):
            data.ctrl[act] = qv
        data.ctrl[self.grip_act] = self.grip_ctrl


# ─────────────────────────────────────────────────────────────────────────────
# Per-arm VR clutch mapper — converts controller pose → target EE pose
# ─────────────────────────────────────────────────────────────────────────────

class VRArmMapper:
    """Origin-relative (clutch) control. Grip press captures the VR origin AND the
    current EE pose; while held the EE target = base_pose + remapped hand delta."""

    def __init__(self, ik: ArmIK, cfg: VRTeleopConfig):
        self.ik = ik
        self.cfg = cfg
        self.reset_anchor()
        self.target_pos = None      # current EE position target (world)
        self.target_quat = None     # current EE orientation target (world, wxyz)

    def reset_anchor(self):
        self.origin_vr_pos = None
        self.origin_vr_quat = None
        self.base_ee_pos = None
        self.base_ee_R = None

    def hold(self, full_qpos: np.ndarray):
        """Freeze the target at the current commanded EE pose (grip released / idle)."""
        pos, R = self.ik.fk(full_qpos)
        self.target_pos = pos
        self.target_quat = mat_to_quat(R)

    def update(self, goal, full_qpos: np.ndarray):
        if goal is None or goal.target_position is None:
            return
        vr_pos = np.asarray(goal.target_position, dtype=np.float64)

        if self.origin_vr_pos is None:
            # Anchor WITHOUT moving: capture VR origin + current EE pose.
            self.origin_vr_pos = vr_pos.copy()
            meta = goal.metadata or {}
            q = meta.get("quaternion")
            self.origin_vr_quat = (
                quat_from_xyzw([q["x"], q["y"], q["z"], q["w"]])
                if q and all(k in q for k in "xyzw") else None)
            self.base_ee_pos, self.base_ee_R = self.ik.fk(full_qpos)
            return

        # Position: remap hand displacement into the robot world frame.
        disp = (vr_pos - self.origin_vr_pos) * self.cfg.pos_scale
        dworld = B_VR_TO_WORLD @ disp
        dworld = dworld * np.asarray(self.cfg.sign)
        self.target_pos = self.base_ee_pos + dworld

        # Orientation: apply the remapped relative hand rotation onto the base pose.
        if self.cfg.track_orientation:
            meta = goal.metadata or {}
            q = meta.get("quaternion")
            if q and all(k in q for k in "xyzw") and self.origin_vr_quat is not None:
                cur = quat_from_xyzw([q["x"], q["y"], q["z"], q["w"]])
                dq_vr = quat_mul(cur, quat_conj(self.origin_vr_quat))  # VR-frame delta
                dq_world = remap_quat_to_world(dq_vr)
                base_q = mat_to_quat(self.base_ee_R)
                self.target_quat = quat_mul(dq_world, base_q)
            else:
                self.target_quat = mat_to_quat(self.base_ee_R)
        else:
            self.target_quat = mat_to_quat(self.base_ee_R)

    def step_ik(self, full_qpos: np.ndarray):
        if self.target_pos is None:
            return
        R_tgt = None
        if self.cfg.track_orientation and self.target_quat is not None:
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, self.target_quat)
            R_tgt = R.reshape(3, 3)
        self.ik.solve(full_qpos, self.target_pos, R_tgt)


# ─────────────────────────────────────────────────────────────────────────────
# Combined HTTP + WebSocket proxy + cloudflared (verbatim port from vr_teleop_sim)
# ─────────────────────────────────────────────────────────────────────────────

_cloudflare_url: str = ""


def make_combined_app(ws_upstream_port: int) -> web.Application:
    async def index(request):
        return web.FileResponse(os.path.join(WEB_UI_DIR, "index.html"))

    async def go(request):
        if _cloudflare_url:
            raise web.HTTPFound(_cloudflare_url)
        return web.Response(text="Tunnel not ready yet, try again in a moment.")

    async def ws_proxy(request):
        client_ws = web.WebSocketResponse()
        await client_ws.prepare(request)
        session = aiohttp.ClientSession()
        try:
            async with session.ws_connect(f"ws://localhost:{ws_upstream_port}") as up:
                async def c2u():
                    async for msg in client_ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await up.send_str(msg.data)
                        elif msg.type == aiohttp.WSMsgType.BINARY:
                            await up.send_bytes(msg.data)
                        elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                            break

                async def u2c():
                    async for msg in up:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await client_ws.send_str(msg.data)
                        elif msg.type == aiohttp.WSMsgType.BINARY:
                            await client_ws.send_bytes(msg.data)
                        elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                            break

                await asyncio.gather(c2u(), u2c(), return_exceptions=True)
        except Exception as e:
            print(f"[WS proxy] {e}")
        finally:
            await session.close()
        return client_ws

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/go", go)
    app.router.add_get("/ws", ws_proxy)
    app.router.add_static("/", WEB_UI_DIR)
    return app


async def start_cloudflared(port: int) -> Optional[subprocess.Popen]:
    cf_bin = "cloudflared"
    try:
        subprocess.run([cf_bin, "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("[cloudflared] Not found — install with: brew install cloudflared")
        print(f"[cloudflared] Then run manually: cloudflared tunnel --url http://localhost:{port}")
        return None

    proc = subprocess.Popen(
        [cf_bin, "tunnel", "--url", f"http://localhost:{port}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    print("[cloudflared] Starting tunnel …")
    url = None
    deadline = time.time() + 30
    while time.time() < deadline:
        line = await asyncio.get_event_loop().run_in_executor(None, proc.stdout.readline)
        if not line:
            break
        m = re.search(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com", line)
        if m:
            url = m.group(0)
            break

    if url:
        global _cloudflare_url
        _cloudflare_url = url
        local_url = f"http://{get_local_ip()}:{port}/go"
        print()
        print("┌──────────────────────────────────────┐")
        print("│  Quest 3 browser → type this:        │")
        print(f"│  {local_url:<36s}  │")
        print("│  It redirects to the VR page; then   │")
        print("│  tap the VR goggles icon & squeeze   │")
        print("│  grip to start controlling.          │")
        print("└──────────────────────────────────────┘")
        print()
    else:
        print("[cloudflared] Could not detect URL — check terminal output")
    return proc


# ─────────────────────────────────────────────────────────────────────────────
# VR backend (runs in a background thread; viewer runs on the main thread)
# ─────────────────────────────────────────────────────────────────────────────

class VRBackend:
    """Spins up the XLeVR websocket server + web proxy + cloudflared in a private
    asyncio loop on a background thread. The main thread polls goals via
    `monitor.get_latest_goal_nowait` (thread-safe)."""

    def __init__(self):
        self.monitor: Optional["VRMonitor"] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=30)
        return self.monitor

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as e:
            print(f"[VRBackend] {e}")

    async def _serve(self):
        monitor = VRMonitor()
        if not monitor.initialize():
            print("[ERROR] Failed to initialize VRMonitor")
            self._ready.set()
            return
        await monitor.vr_server.start()
        monitor.is_running = True
        self.monitor = monitor

        ws_port = monitor.config.websocket_port
        runner = web.AppRunner(make_combined_app(ws_port))
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", COMBINED_PORT).start()
        print(f"🌐 VR server on port {COMBINED_PORT}")

        cf_proc = await start_cloudflared(COMBINED_PORT)
        monitor_task = asyncio.create_task(monitor.monitor_commands())
        self._ready.set()

        try:
            while not self._stop.is_set():
                await asyncio.sleep(0.1)
        finally:
            monitor_task.cancel()
            await monitor.vr_server.stop()
            await runner.cleanup()
            if cf_proc:
                cf_proc.terminate()
            print("[VRBackend] stopped.")

    def stop(self):
        self._stop.set()


# ─────────────────────────────────────────────────────────────────────────────
# Main control loop
# ─────────────────────────────────────────────────────────────────────────────

def grip_active(goal) -> bool:
    return bool((goal.metadata or {}).get("grip_active", False)) if goal else False


def trigger_value(goal) -> float:
    return float((goal.metadata or {}).get("trigger", 0.0)) if goal else 0.0


def collect_free_object_state(model, data):
    """Find every free-joint body (the test blocks added to station.xml) and
    capture the qpos/qvel slices + initial qpos so they can be reset on demand.
    The arms have no free joints, so this returns exactly the test objects."""
    qpos_slices, dof_slices = [], []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            qadr = int(model.jnt_qposadr[j])
            dadr = int(model.jnt_dofadr[j])
            qpos_slices.append((qadr, qadr + 7))   # free joint: 7 qpos (xyz + wxyz)
            dof_slices.append((dadr, dadr + 6))     #            6 qvel (lin + ang)
    return qpos_slices, dof_slices, data.qpos.copy()


def reset_test_objects(model, data, qpos_slices, dof_slices, init_qpos):
    """Teleport the free-joint test objects back to their start pose with zero
    velocity, leaving the arms untouched. Bound to ENTER in the viewer window."""
    for a, b in qpos_slices:
        data.qpos[a:b] = init_qpos[a:b]
    for a, b in dof_slices:
        data.qvel[a:b] = 0.0
    mujoco.mj_forward(model, data)
    print(f"[Reset] {len(qpos_slices)} test object(s) returned to start positions.")


def home_arms(model, data, iks, arms, cfg, viewer=None):
    """Smoothly ramp the controlled arms to HOME_QPOS (cosine ease-in-out) while
    stepping physics, so a double-click grip resets the robot the same way
    xlerobot's `_home_arms` does. Opens the grippers on the way home."""
    n = max(1, int(cfg.home_seconds * cfg.control_hz))
    starts = {a: iks[a].q_cmd.copy() for a in arms}
    print(f"[Home] ramping {', '.join(arms)} to home over "
          f"{cfg.home_seconds:.1f}s — keep the workspace clear …")
    for i in range(1, n + 1):
        s = 0.5 - 0.5 * math.cos(math.pi * i / n)   # ease in-out
        for a in arms:
            iks[a].q_cmd = (1.0 - s) * starts[a] + s * HOME_QPOS
            iks[a].grip_ctrl = GRIP_OPEN_CTRL
            iks[a].apply(data)
        for _ in range(cfg.physics_substeps):
            mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        time.sleep(1.0 / cfg.control_hz)
    print("[Home] at rest pose.")


def run(cfg: VRTeleopConfig):
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Test-object reset: capture the blocks' start pose and bind ENTER (in the
    # viewer window) to teleport them back. The passive viewer has no custom
    # on-screen buttons, so a key press is the "reset button".
    obj_qpos_slices, obj_dof_slices, obj_init_qpos = collect_free_object_state(model, data)
    reset_request = threading.Event()

    def key_callback(keycode):
        if keycode == 257:        # GLFW_KEY_ENTER → reset test objects
            reset_request.set()

    arms = ["left", "right"] if cfg.arm == "both" else [cfg.arm]

    iks = {s: ArmIK(model, s, cfg) for s in ARM_SPEC}
    mappers = {s: VRArmMapper(iks[s], cfg) for s in ARM_SPEC}
    for s in ARM_SPEC:
        iks[s].seed_from(data.qpos)
        mappers[s].hold(data.qpos)
        iks[s].apply(data)

    # ── VR backend ──
    backend = VRBackend()
    if VRMonitor is None:
        print("[ERROR] VRMonitor unavailable; cannot start VR. Use --test for a dry run.")
        return
    monitor = backend.start()
    if monitor is None:
        print("[ERROR] VR backend failed to start.")
        return

    print(f"\n[Info] arms={arms}  hz={cfg.control_hz}  "
          f"orientation={'on' if cfg.track_orientation else 'off'}")
    print("[Info] Arm holds its pose until you SQUEEZE GRIP.")
    print("[Info]   • grip ON  → anchors at the current pose; your hand then drives the")
    print("[Info]     gripper: push fwd → reach, move L/R/up/down, rotate → wrist.")
    print("[Info]   • grip OFF → arm freezes; reposition your hand and grip again.")
    print("[Info]   • DOUBLE-CLICK GRIP → return to the home pose and start over.")
    print("[Info]   • trigger  → close/open gripper.")
    print("[Info]   • press ENTER in the MuJoCo window → reset the test blocks.")
    print("[Info]   Close the viewer window to stop.\n")

    prev_grip = {a: False for a in arms}
    last_press = {a: -1e9 for a in arms}   # last grip rising-edge time (double-click)
    dt = 1.0 / cfg.control_hz

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        try:
            while viewer.is_running():
                t0 = time.perf_counter()

                if reset_request.is_set():
                    reset_request.clear()
                    reset_test_objects(model, data, obj_qpos_slices,
                                       obj_dof_slices, obj_init_qpos)

                goals = {a: monitor.get_latest_goal_nowait(a) for a in arms}
                g_now = {a: grip_active(goals[a]) for a in arms}

                # Double-click grip on any controller → home + fresh start.
                restart = False
                for a in arms:
                    if g_now[a] and not prev_grip[a]:           # grip rising edge
                        if (t0 - last_press[a]) < DOUBLE_CLICK_WINDOW:
                            restart = True
                        last_press[a] = t0

                if restart:
                    print("[Restart] grip double-click → homing and starting over.")
                    home_arms(model, data, iks, arms, cfg, viewer)
                    for a in arms:
                        iks[a].seed_from(data.qpos)
                        mappers[a].reset_anchor()
                        mappers[a].hold(data.qpos)
                        prev_grip[a] = g_now[a]   # current (held) state → no spurious re-anchor
                        last_press[a] = -1e9
                    time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
                    continue

                for a in arms:
                    g = g_now[a]
                    if g and not prev_grip[a]:
                        mappers[a].reset_anchor()
                        iks[a].seed_from(data.qpos)   # start IK from where the arm actually is
                        print(f"[{a}] grip ON — anchored.")
                    elif prev_grip[a] and not g:
                        mappers[a].hold(data.qpos)
                        print(f"[{a}] grip OFF — frozen.")

                    if g:
                        mappers[a].update(goals[a], data.qpos)
                        mappers[a].step_ik(data.qpos)
                        # gripper: trigger pressed → close
                        t = trigger_value(goals[a])
                        iks[a].grip_ctrl = (GRIP_CLOSED_CTRL
                                            if t > cfg.gripper_trigger_threshold
                                            else GRIP_OPEN_CTRL)
                    prev_grip[a] = g
                    iks[a].apply(data)

                for _ in range(cfg.physics_substeps):
                    mujoco.mj_step(model, data)
                viewer.sync()

                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
        except KeyboardInterrupt:
            pass
        finally:
            backend.stop()
            print("[Sim] Stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# Headless-VR self-test: drive the left EE in a small circle to validate IK+viewer
# ─────────────────────────────────────────────────────────────────────────────

def run_test(cfg: VRTeleopConfig):
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    side = cfg.arm if cfg.arm in ("left", "right") else "left"
    ik = ArmIK(model, side, cfg)
    ik.seed_from(data.qpos)
    p0, R0 = ik.fk(data.qpos)
    print(f"[test] {side} EE start pos = {np.round(p0, 3)}")

    obj_qpos_slices, obj_dof_slices, obj_init_qpos = collect_free_object_state(model, data)
    reset_request = threading.Event()

    def key_callback(keycode):
        if keycode == 257:        # GLFW_KEY_ENTER → reset test objects
            reset_request.set()

    print("[test] Driving the EE in a 8cm circle (position-only). "
          "Press ENTER to reset test blocks; close the viewer to stop.")
    t0 = time.time()
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            if reset_request.is_set():
                reset_request.clear()
                reset_test_objects(model, data, obj_qpos_slices,
                                   obj_dof_slices, obj_init_qpos)
            t = time.time() - t0
            tgt = p0 + np.array([0.0, 0.06 * math.sin(t), 0.06 * math.cos(t) - 0.06])
            ik.solve(data.qpos, tgt, R0 if cfg.track_orientation else None)
            ik.apply(data)
            for _ in range(cfg.physics_substeps):
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(1.0 / cfg.control_hz)


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="VR teleop: Meta Quest 3 → YAM station MuJoCo sim")
    p.add_argument("--arm", default="both", choices=["left", "right", "both"],
                   help="Which arm to control (default: both)")
    p.add_argument("--control_hz", type=float, default=60.0)
    p.add_argument("--pos-scale", type=float, default=1.0,
                   help="Hand metres → robot metres (default 1.0; <1 = finer)")
    p.add_argument("--no-orientation", dest="track_orientation", action="store_false",
                   help="Position-only: gripper keeps its grip-time orientation.")
    p.add_argument("--home_seconds", type=float, default=1.5,
                   help="Duration of the double-click-grip homing ramp (default 1.5s).")
    p.add_argument("--test", action="store_true",
                   help="No headset: drive the EE in a circle to validate IK + viewer.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = VRTeleopConfig(arm=args.arm, control_hz=args.control_hz,
                         pos_scale=args.pos_scale,
                         home_seconds=args.home_seconds,
                         track_orientation=args.track_orientation)
    if args.test:
        run_test(cfg)
    else:
        run(cfg)
