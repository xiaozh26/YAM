#!/usr/bin/env python3
"""
Vendored VRMonitor for the YAM station VR teleop.

Self-contained copy of the XLeVR VRMonitor, trimmed for the MuJoCo teleop:
  • no hardcoded XLEVR_PATH and no os.chdir — this package lives next to xlevr/
  • no SimpleHTTPSServer (the teleop serves web-ui itself over HTTP + cloudflared,
    and the websocket server runs plain ws with ssl=None)

It exposes the same interface the control loop relies on:
    monitor = VRMonitor(); monitor.initialize(); await monitor.vr_server.start()
    monitor.is_running = True; asyncio.create_task(monitor.monitor_commands())
    goal = monitor.get_latest_goal_nowait("left" | "right" | "headset")
"""

import asyncio
import socket
import threading

# xlevr/ sits beside this file; the teleop script adds this dir to sys.path.
from xlevr.config import XLeVRConfig
from xlevr.inputs.vr_ws_server import VRWebSocketServer
from xlevr.inputs.base import ControlGoal, ControlMode  # noqa: F401 (re-exported)


def get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "localhost"


class VRMonitor:
    """Thread-safe wrapper around the XLeVR websocket server.

    After initialize() + vr_server.start() + monitor_commands() (in an asyncio
    loop), poll get_latest_goal_nowait() from any thread.
    """

    def __init__(self):
        self.config = None
        self.vr_server = None
        self.is_running = False
        self.command_queue = None
        self.latest_goal = None
        self.left_goal = None
        self.right_goal = None
        self.headset_goal = None
        self._goal_lock = threading.Lock()

    def initialize(self) -> bool:
        print("🔧 Initializing VR Monitor...")
        self.config = XLeVRConfig()
        self.config.enable_vr = True
        self.config.enable_keyboard = False
        self.config.enable_https = False

        self.command_queue = asyncio.Queue()
        try:
            self.vr_server = VRWebSocketServer(
                command_queue=self.command_queue,
                config=self.config,
                print_only=False,
            )
        except Exception as e:
            print(f"❌ Failed to create VR WebSocket server: {e}")
            return False

        print("✅ VR Monitor initialized")
        return True

    async def monitor_commands(self):
        print("📊 Monitoring VR control commands...")
        while self.is_running:
            try:
                goal = await asyncio.wait_for(self.command_queue.get(), timeout=1.0)
                with self._goal_lock:
                    if goal.arm == "left":
                        self.left_goal = goal
                    elif goal.arm == "right":
                        self.right_goal = goal
                    elif goal.arm == "headset":
                        self.headset_goal = goal
                    self.latest_goal = goal
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"❌ Error processing command: {e}")

    def get_latest_goal_nowait(self, arm=None):
        """Poll the latest goal without blocking.

        arm="left"/"right"/"headset" → that controller's ControlGoal (or None);
        arm=None → dict with all three plus has_* flags.
        """
        with self._goal_lock:
            if arm == "left":
                return self.left_goal
            elif arm == "right":
                return self.right_goal
            elif arm == "headset":
                return self.headset_goal
            return {
                "left": self.left_goal,
                "right": self.right_goal,
                "headset": self.headset_goal,
                "has_left": self.left_goal is not None,
                "has_right": self.right_goal is not None,
                "has_headset": self.headset_goal is not None,
            }
