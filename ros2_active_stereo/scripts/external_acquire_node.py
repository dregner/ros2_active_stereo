#!/usr/bin/env python3
"""
phase_processing_node.py
========================
Unified ROS 2 node for structured-light active-stereo fringe projection.

Architecture
------------
The heavy lifting (projector control, camera triggering, fringe/GrayCode
generation, and phase computation) is done by the standalone C++ binary
``stereo_fringe_main``, which runs **outside** ROS to minimise
projection/acquisition latency and avoid shadows from ROS message passing.

This node's responsibilities:
  1. IDLE  – wait for acquisition requests.
  2. /run_acquisition  – send ACQUIRE to the C++ binary.
  3. Load the EXR result files written by the binary:
        <output_dir>/phase_map_left.exr
        <output_dir>/phase_map_right.exr
        <output_dir>/modulation_left.exr
        <output_dir>/modulation_right.exr
  4. Publish them as ROS images on standard topics.
  5. Return to IDLE.
"""

import os
import subprocess
import threading
import time

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger
from std_srvs.srv import SetBool
from fringe_process import FringeProcess   # type: ignore

from ros2_active_stereo_msgs.srv import ProcessFolder  # type: ignore


# ── Node ──────────────────────────────────────────────────────────────────────

class ProcessPhase(Node):
    """Acquisition-handoff + phase-result publisher node."""

    def __init__(self):
        super().__init__("process_phase")
        self.get_logger().info("Node 'process_phase' initialising …")

        self._bridge = CvBridge()
        self._pipeline_lock = threading.Lock()
        self.stereo_processor = None  # type: ignore

        # State ────────────────────────────────────────────────────────────────
        self._state_pub = self.create_publisher(String, "state_process_phase", 10)
        self._current_state = "UNCONFIGURED"
        self._publish_state(self._current_state)
        self.create_timer(0.5, self._publish_state_periodically)

        # Parameters ───────────────────────────────────────────────────────────
        self.declare_parameter(
            "binary_path",
            f"/home/{os.getenv('USER')}/fringe-projection/build/stereo_fringe")
        self.declare_parameter(
            "config_path",
            f"/home/{os.getenv('USER')}/fringe-projection/config/stereo_config.yaml")
        self.declare_parameter("use_gpio",         True)
        self.declare_parameter("save_raw_frames",  True)
        self.declare_parameter("binary_timeout_s", 120.0)

        # fringe_capture section (mirrored in YAML)
        self.declare_parameter("pixels_per_fringe",      64)
        self.declare_parameter("n_steps",                  8)
        self.declare_parameter("projector_display_ms",    50)
        self.declare_parameter("projector_monitor_name", "DP-1")
        self.declare_parameter("projector_window_name",  "Projector")
        self.declare_parameter("proj_width",            1920)
        self.declare_parameter("proj_height",           1080)
        self.declare_parameter("cam_width",              2448)
        self.declare_parameter("cam_height",             2048)

        self.declare_parameter("zncc_n_imgs", 5)
        self.declare_parameter("zncc_steps", 20)
        self.declare_parameter("zncc_warminig", 0)
        # Debug
        self.declare_parameter("debug_show", True)

        # Publishers ───────────────────────────────────────────────────────────
        self._pub_phi_left   = self.create_publisher(Image, "sync/left/phase_map",       10)
        self._pub_phi_right  = self.create_publisher(Image, "sync/right/phase_map",      10)
        self._pub_mod_left   = self.create_publisher(Image, "sync/left/modulation_map",  10)
        self._pub_mod_right  = self.create_publisher(Image, "sync/right/modulation_map", 10)
        self._pub_dbg_left   = self.create_publisher(Image, "sync/left/debug/phase_map",  10)
        self._pub_dbg_right  = self.create_publisher(Image, "sync/right/debug/phase_map", 10)

        # Services ─────────────────────────────────────────────────────────────
        self.create_service(ProcessFolder, "run_acquisition", self._fringe_acquisition_callback)
        self.create_service(Trigger,       "reconfigure",     self._reconfigure_callback)
        self.create_service(SetBool,       "correlation_process", self._laser_acquisition_callback)

        self._proc = None
        self._acquire_done_event = threading.Event()
        self._start_cpp_process()

        self._publish_state("IDLE")
        self.get_logger().info("process_phase node ready.")

    def __del__(self):
        self._stop_cpp_process()

    def _start_cpp_process(self):
        p = self._read_params()
        binary = p["binary_path"]
        
        cmd = [binary, "--config", p["config_path"]]
        if not p["use_gpio"]:
            cmd.append("--no-gpio")
        if p["save_raw_frames"]:
            cmd.append("--save-raw")

        self.get_logger().info(f"Starting C++ backend: {' '.join(cmd)}")

        env = os.environ.copy()
        for k in list(env.keys()):
            if k.startswith("QT_QPA") or k == "QT_PLUGIN_PATH":
                del env[k]

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1
        )

        def log_stdout():
            if self._proc and self._proc.stdout:
                for line in self._proc.stdout:
                    clean_line = line.rstrip()
                    self.get_logger().info(f"[stereo_fringe] {clean_line}")
                    if "[Pipeline] ACQUIRE_DONE" in clean_line or "[Pipeline] ZNCC_DONE" in clean_line:
                        self._acquire_done_event.set()
        
        threading.Thread(target=log_stdout, daemon=True).start()

    def _stop_cpp_process(self):
        if self._proc:
            try:
                self._proc.stdin.write("QUIT\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=3.0)
            except Exception:
                self._proc.kill()
            self._proc = None

    def _publish_state(self, state: str) -> None:
        self._current_state = state
        msg = String()
        msg.data = state
        self._state_pub.publish(msg)
        self.get_logger().info(f"[State] {state}")

    def _publish_state_periodically(self) -> None:
        msg = String()
        msg.data = self._current_state
        self._state_pub.publish(msg)

    def _read_params(self) -> dict:
        return dict(
            binary_path            = str(self.get_parameter("binary_path").value),
            config_path            = str(self.get_parameter("config_path").value),
            use_gpio               = bool(self.get_parameter("use_gpio").value),
            save_raw_frames        = bool(self.get_parameter("save_raw_frames").value),
            binary_timeout_s       = float(self.get_parameter("binary_timeout_s").value),
            pixels_per_fringe      = int(self.get_parameter("pixels_per_fringe").value),
            n_steps                = int(self.get_parameter("n_steps").value),
            projector_display_ms   = int(self.get_parameter("projector_display_ms").value),
            projector_monitor_name = str(self.get_parameter("projector_monitor_name").value),
            projector_window_name  = str(self.get_parameter("projector_window_name").value),
            proj_width             = int(self.get_parameter("proj_width").value),
            proj_height            = int(self.get_parameter("proj_height").value),
            cam_width              = int(self.get_parameter("cam_width").value),
            cam_height             = int(self.get_parameter("cam_height").value),
            debug_show             = bool(self.get_parameter("debug_show").value),
            zncc_n_img             = int(self.get_parameter("zncc_n_imgs").value),
            zncc_steps             = int(self.get_parameter("zncc_steps").value),
            zncc_warmup             = int(self.get_parameter("zncc_warminig").value)
        )

    def _write_yaml(self, p: dict, output_dir: str) -> None:
        config_path = p["config_path"]
        existing: dict = {}
        if os.path.isfile(config_path):
            with open(config_path, "r") as fh:
                existing = yaml.safe_load(fh) or {}

        existing["fringe_capture"] = {
            "projector_resolution": {
                "width":  p["proj_width"],
                "height": p["proj_height"],
            },
            "camera_resolution": {
                "width":  p["cam_width"],
                "height": p["cam_height"],
            },
            "pixels_per_fringe":      p["pixels_per_fringe"],
            "n_steps":                p["n_steps"],
            "projector_display_ms":   p["projector_display_ms"],
            "projector_window_name":  p["projector_window_name"],
            "projector_monitor_name": p["projector_monitor_name"],
            "output_dir":             output_dir,
            "save_raw_frames":        p["save_raw_frames"],
            "zncc": {
                "num_images":           p["zncc_n_img"],
                "motor_steps":          p["zncc_steps"],
                "warmup_triggers":      p["zncc_warmup"],
            }
        }

        with open(config_path, "w") as fh:
            yaml.dump(existing, fh, default_flow_style=False, allow_unicode=True)

        self.get_logger().info(f"YAML updated: {config_path!r}  (output_dir={output_dir!r})")

    def _reconfigure_callback(self, _request, response: Trigger.Response):
        try:
            p = self._read_params()
            cfg_path = p["config_path"]
            current_output_dir = f"/home/{os.getenv('USER')}/active_results"
            if os.path.isfile(cfg_path):
                with open(cfg_path, "r") as fh:
                    existing = yaml.safe_load(fh) or {}
                fc = existing.get("fringe_capture", {})
                current_output_dir = fc.get("output_dir", current_output_dir)

            self._write_yaml(p, current_output_dir)
            
            if self._proc is None or self._proc.poll() is not None:
                self._start_cpp_process()
                
            response.success = True
            response.message = f"YAML updated: {cfg_path}"
            self.get_logger().info("Reconfigured: YAML written.")
        except Exception as exc:
            self.get_logger().error(f"Reconfigure error: {exc}")
            response.success = False
            response.message = str(exc)
        return response


    def _laser_acquisition_callback(self, request: SetBool.Request, response: SetBool.Response):
        if not request.data:
            response.success = False
            response.message = "Ignored"
            return response
            
        if not self._pipeline_lock.acquire(blocking=False):
            response.success = False
            response.message = "Acquisition already running."
            return response

        def worker():
            try:
                p = self._read_params()
                out_dir = f"/tmp/rrp_stereo"
                os.makedirs(out_dir, exist_ok=True)
                
                if self._proc is None or self._proc.poll() is not None:
                    self._start_cpp_process()

                self._publish_state("ZNCC_ACQUIRING")
                
                num_images = p["zncc_n_img"]
                steps = p["zncc_steps"]
                warmup_triggers = p["zncc_warmup"]
                
                self._acquire_done_event.clear()
                if self._proc and self._proc.stdin:
                    self._proc.stdin.write(f"LASER {out_dir}\n")
                    self._proc.stdin.flush()
                
                if not self._acquire_done_event.wait(timeout=120.0):
                    raise RuntimeError("C++ ZNCC binary timed out.")
                
                self.get_logger().info("C++ ZNCC acquisition completed successfully.")
            except Exception as exc:
                self.get_logger().error(f"ZNCC Pipeline error: {exc}")
                if self._proc and self._proc.poll() is not None:
                     self._proc = None
            finally:
                self._pipeline_lock.release()
                self._publish_state("IDLE")

        threading.Thread(target=worker, daemon=True).start()
        response.success = True
        response.message = "ZNCC Acquisition started."
        return response

    def _fringe_acquisition_callback(self, request: ProcessFolder.Request, response: ProcessFolder.Response):
        folder = request.folder_path.strip() or f"/home/{os.getenv('USER')}/active_results"
        self.get_logger().info(f"/run_acquisition called → output_dir={folder!r}")

        if not self._pipeline_lock.acquire(blocking=False):
            response.success = False
            response.message = "Acquisition already running."
            return response

        t = threading.Thread(target=self._pipeline_worker, args=(folder,), daemon=True)
        t.start()

        response.success = True
        response.message = "Acquisition started."
        return response

    def _pipeline_worker(self, output_dir: str) -> None:
        try:
            p = self._read_params()
            self.stereo_processor = FringeProcess(img_resolution=(p["proj_width"], p["proj_height"]), camera_resolution=(p["cam_width"], p["cam_height"]), px_f=p["pixels_per_fringe"], steps=p["n_steps"])
            
            self._publish_state("CONFIGURING")
            os.makedirs(output_dir, exist_ok=True)
            self._write_yaml(p, output_dir)
            
            if self._proc is None or self._proc.poll() is not None:
                self._start_cpp_process()

            self._publish_state("ACQUIRING")
            
            self._acquire_done_event.clear()
            if self._proc and self._proc.stdin:
                self._proc.stdin.write(f"FRINGE {output_dir}\n")
                self._proc.stdin.flush()
            
            # Wait for acquisition to finish
            if not self._acquire_done_event.wait(timeout=p["binary_timeout_s"]):
                raise RuntimeError(f"C++ binary timed out after {p['binary_timeout_s']} s.")
            
            self.get_logger().info("C++ acquisition completed successfully.")

            self._publish_state("PROCESSING")
            left_images = sorted(os.listdir(os.path.join(output_dir,'fringe', "left")))
            right_images = sorted(os.listdir(os.path.join(output_dir,'fringe', "right")))
            for idx, (left, right) in enumerate(zip(left_images, right_images)):
                self.stereo_processor.set_images(image_left=self.load_image(os.path.join(output_dir,'fringe', "left", left)),
                                               image_right=self.load_image(os.path.join(output_dir,'fringe', "right", right)),
                                               counter=idx)

            abs_phi_l, abs_phi_r, mod_l, mod_r = self.stereo_processor.calculate_abs_phi_images()

            stamp = self.get_clock().now().to_msg()

            self._publish_image(self._pub_phi_left,  abs_phi_l.astype(np.float64), "64FC1", stamp, "Active/left_camera_link")
            self._publish_image(self._pub_phi_right, abs_phi_r.astype(np.float64), "64FC1", stamp, "Active/right_camera_link")
            self._publish_image(self._pub_mod_left,  self._norm_u8(mod_l), "8UC1", stamp, "Active/left_camera_link")
            self._publish_image(self._pub_mod_right, self._norm_u8(mod_r), "8UC1", stamp, "Active/right_camera_link")

            if p["debug_show"]:
                self._publish_debug_image(self._pub_dbg_left,  abs_phi_l, stamp)
                self._publish_debug_image(self._pub_dbg_right, abs_phi_r, stamp)

            self.get_logger().info("Results published. Returning to IDLE.")

        except Exception as exc:
            self.get_logger().error(f"Pipeline error: {exc}")
            self._publish_state("ERROR")
            if self._proc and self._proc.poll() is not None:
                 self._proc = None # Reset process if it crashed
        finally:
            self._pipeline_lock.release()
            self._publish_state("IDLE")

    @staticmethod
    def load_image(path: str) -> np.ndarray:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Expected image not found: {path!r}")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"cv2.imread failed for {path!r}")
        if img.ndim == 3 and img.shape[2] == 1:
            img = img[:, :, 0]
        return img

    @staticmethod
    def _norm_u8(img: np.ndarray) -> np.ndarray:
        mn, mx = float(np.nanmin(img)), float(np.nanmax(img))
        if mx > mn:
            return ((img - mn) / (mx - mn) * 255).astype(np.uint8)
        return np.zeros_like(img, dtype=np.uint8)

    def _publish_image(
        self,
        publisher,
        image: np.ndarray,
        encoding: str,
        stamp,
        frame_id: str = "Active/left_camera_link",
    ) -> None:
        try:
            msg = self._bridge.cv2_to_imgmsg(image, encoding=encoding)
            msg.header.stamp    = stamp
            msg.header.frame_id = frame_id
            publisher.publish(msg)
        except Exception as exc:
            self.get_logger().error(f"Publish error ({encoding}): {exc}")

    def _publish_debug_image(
        self,
        publisher,
        image_f32: np.ndarray,
        stamp,
        frame_id: str = "Active/left_camera_link",
    ) -> None:
        vis = self._norm_u8(image_f32)
        try:
            msg = self._bridge.cv2_to_imgmsg(vis, encoding="mono8")
            msg.header.stamp    = stamp
            msg.header.frame_id = frame_id
            publisher.publish(msg)
        except Exception as exc:
            self.get_logger().error(f"Debug publish error: {exc}")


def main(args=None):
    rclpy.init(args=args)
    node = ProcessPhase()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
