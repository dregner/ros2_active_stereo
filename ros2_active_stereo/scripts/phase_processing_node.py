#!/usr/bin/env python3
"""
phase_processing_node.py
========================
ROS 2 node that:
  1. Projects fringe + Gray Code patterns via ProjectorControl.
  2. Receives hardware-triggered stereo image pairs.
  3. Computes absolute phase maps using FringeProcess (GPU/CPU via PyTorch).
  4. Publishes phase maps and modulation maps for the triangulation node.

Key design decisions
--------------------
* The service callback (restart_phase_callback) returns immediately — the caller
  is never blocked waiting for projection/acquisition to finish.
* Heavy phase computation (calculate_abs_phi_images) runs in a daemon thread so
  the ROS executor thread is never stalled.
* The projector timer is created once per scan step and cancelled after firing,
  preventing duplicate timer accumulation.
* cv2.waitKey uses 1 ms (minimum) so the OpenCV window stays responsive without
  introducing artificial delays.
"""

import os
import threading
import rclpy
import time
import cv2
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from std_msgs.msg import String
import message_filters
from cv_bridge import CvBridge
from fringe_process import FringeProcess   # type: ignore
from projector import ProjectorControl     # type: ignore


class ProcessPhase(Node):
    def __init__(self):
        super().__init__('process_phase')
        self.get_logger().info("Node 'process_phase' starting…")

        # ── State ─────────────────────────────────────────────────────────
        self.bridge = CvBridge()
        self.control_process: bool = False
        self.index: int = 0
        self.index_max: int = 0
        self._process_thread: threading.Thread | None = None

        # Timer handle — never allow duplicates
        self.trigger_timer = None
        self._timer_lock = threading.Lock()

        # ── State publisher ────────────────────────────────────────────────
        self.state_pub = self.create_publisher(String, 'state_process_phase', 10)
        self.current_state = 'UNCONFIGURED'
        self._publish_state(self.current_state)
        self.state_timer = self.create_timer(0.5, self._publish_state_periodically)

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter('px_f',            32)
        self.declare_parameter('steps',            8)
        self.declare_parameter('index',            0)
        self.declare_parameter('control_process',  False)
        self.declare_parameter('hz',               5)
        self.declare_parameter('debug_save',       True)
        self.declare_parameter('debug_show',       True)
        self.declare_parameter('a',                1)

        # ── Configure ─────────────────────────────────────────────────────
        self._on_configure()

    # ──────────────────────────────────────────────────────────────────────
    # State helpers
    # ──────────────────────────────────────────────────────────────────────
    def _publish_state(self, state: str):
        self.current_state = state
        msg = String()
        msg.data = state
        self.state_pub.publish(msg)
        self.get_logger().info(f"[State] {state}")

    def _publish_state_periodically(self):
        msg = String()
        msg.data = self.current_state
        self.state_pub.publish(msg)

    # ──────────────────────────────────────────────────────────────────────
    # Configuration / activation
    # ──────────────────────────────────────────────────────────────────────
    def _on_configure(self):
        try:
            self._publish_state('CONFIGURING')

            self.control_process = self.get_parameter('control_process').value
            self.debug_save      = self.get_parameter('debug_save').value
            self.debug_show      = self.get_parameter('debug_show').value
            self.px_f            = self.get_parameter('px_f').value
            self.steps           = self.get_parameter('steps').value
            self.index           = self.get_parameter('index').value
            self.hz              = self.get_parameter('hz').value

            # Projector + fringe processor
            self.projector = ProjectorControl(index=self.index)
            self.get_logger().info(f"Projector resolution: {self.projector.img_resolution}")

            self.stereo_processor = FringeProcess(
                img_resolution=self.projector.img_resolution,
                px_f=self.px_f,
                steps=self.steps)

            # Build LUT and load images into projector
            a = self.get_parameter('a').value
            self.lut = np.clip((np.arange(256)) / a, 0, 255).astype(np.uint8)
            fringe_images   = cv2.LUT(self.stereo_processor.get_fr_image(), self.lut)
            graycode_images = self.stereo_processor.get_gc_images()
            self.projector.set_images(fringe_images, graycode_images)
            self.projector.setup_projector_window()

            # Image subscribers (synchronized)
            self.sm4_left_sub  = message_filters.Subscriber(self, Image, 'sync/left/image_raw')
            self.sm4_right_sub = message_filters.Subscriber(self, Image, 'sync/right/image_raw')

            # Publishers
            self.abs_phi_left_pub  = self.create_publisher(Image, 'sync/left/phase_map',       10)
            self.abs_phi_right_pub = self.create_publisher(Image, 'sync/right/phase_map',      10)
            self.mask_left_pub     = self.create_publisher(Image, 'sync/left/modulation_map',  10)
            self.mask_right_pub    = self.create_publisher(Image, 'sync/right/modulation_map', 10)

            self.abs_phi_left_debug_pub  = self.create_publisher(Image, 'sync/left/debug/phase_map',  10)
            self.abs_phi_right_debug_pub = self.create_publisher(Image, 'sync/right/debug/phase_map', 10)

            # Trigger client
            self.trigger_client = self.create_client(Trigger, 'trigger')
            while not self.trigger_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('Waiting for trigger service…')

            # Phase process service — returns immediately
            self.restart_phase_service = self.create_service(
                Trigger, 'phase_process', self._restart_phase_callback)

            self._publish_state('CONFIGURED')
            self._on_activate()

        except Exception as e:
            self.get_logger().error(f'Configuration error: {e}')
            self._publish_state('ERROR_IN_CONFIGURATION')

    def _on_activate(self):
        try:
            self._publish_state('ACTIVATING')
            self.ts = message_filters.ApproximateTimeSynchronizer(
                [self.sm4_left_sub, self.sm4_right_sub],
                queue_size=5, slop=0.05)
            self.ts.registerCallback(self._synced_callback)
            self._publish_state('ACTIVE')
        except Exception as e:
            self.get_logger().error(f'Activation error: {e}')
            self._publish_state('ERROR_IN_ACTIVATION')

    # ──────────────────────────────────────────────────────────────────────
    # Service callback — returns immediately so the caller never blocks
    # ──────────────────────────────────────────────────────────────────────
    def _restart_phase_callback(self, request, response):
        """Start a new acquisition cycle. Returns immediately."""
        self.get_logger().info('Phase process requested.')

        # Re-read parameters in case they were changed
        self.control_process = self.get_parameter('control_process').value
        self.debug_save      = self.get_parameter('debug_save').value
        self.debug_show      = self.get_parameter('debug_show').value
        self.px_f            = self.get_parameter('px_f').value
        self.steps           = self.get_parameter('steps').value
        self.hz              = self.get_parameter('hz').value

        # Respond immediately so the service client is not blocked
        response.success = True
        response.message = 'Acquisition started'

        # Kick off the scan
        self.control_process = True
        self.projector.current_image_index = -1
        self._publish_state('RESTARTING_PROCESSING')
        self._change_index()

        return response

    # ──────────────────────────────────────────────────────────────────────
    # Image callback
    # ──────────────────────────────────────────────────────────────────────
    def _synced_callback(self, left_msg, right_msg):
        if not self.control_process:
            return

        image_left  = self.bridge.imgmsg_to_cv2(left_msg,  'mono8')
        image_right = self.bridge.imgmsg_to_cv2(right_msg, 'mono8')
        self._process_images(image_left, image_right)
        self._change_index()

    # ──────────────────────────────────────────────────────────────────────
    # Image processing
    # ──────────────────────────────────────────────────────────────────────
    def _process_images(self, image_left: np.ndarray, image_right: np.ndarray):
        self.get_logger().info(f"Storing image {self.index}/{self.index_max}")
        self.stereo_processor.set_images(image_left, image_right, self.index)

        if self.debug_save:
            self._debug_save_images(image_left, image_right, self.index)

        if self.index == self.index_max:
            # All images acquired — compute phase maps in a background thread
            # so the ROS executor is NOT blocked.
            self.get_logger().info("Last image received. Starting phase computation thread…")
            self.control_process = False
            # Project black while computing
            self.projector.project_image(
                np.zeros((*self.projector.img_resolution[::-1], 3), dtype=np.uint8))
            self._launch_processing_thread()

    def _launch_processing_thread(self):
        """Spawn a daemon thread for the heavy phase computation."""
        if self._process_thread is not None and self._process_thread.is_alive():
            self.get_logger().warn("Previous processing thread still running — skipping.")
            return
        self._process_thread = threading.Thread(
            target=self._processing_worker, daemon=True)
        self._process_thread.start()

    def _processing_worker(self):
        """Heavy computation — runs off the ROS executor thread."""
        try:
            self._publish_state('PROCESSING')
            abs_phi_l, abs_phi_r, mod_l, mod_r = \
                self.stereo_processor.calculate_abs_phi_images()

            if self.debug_save:
                np.save('abs_phi_left.npy',  abs_phi_l)
                np.save('abs_phi_right.npy', abs_phi_r)
                np.save('mask_left.npy',     mod_l)
                np.save('mask_right.npy',    mod_r)
                self.get_logger().info("Phase maps saved as .npy files.")

            # Publish results
            self._publish_image(self.abs_phi_left_pub,  abs_phi_l, '64FC1')
            self._publish_image(self.abs_phi_right_pub, abs_phi_r, '64FC1')
            self._publish_image(self.mask_left_pub,     mod_l,     '8UC1')
            self._publish_image(self.mask_right_pub,    mod_r,     '8UC1')

            if self.debug_show:
                self._publish_debug_image(self.abs_phi_left_debug_pub,  abs_phi_l)
                self._publish_debug_image(self.abs_phi_right_debug_pub, abs_phi_r)

            self._publish_state('FINISHED_PROCESSING')
            self.get_logger().info("Phase maps published successfully.")

        except Exception as e:
            self.get_logger().error(f'Phase computation error: {e}')
            self._publish_state('ERROR_IN_PROCESSING')

    # ──────────────────────────────────────────────────────────────────────
    # Projection / triggering
    # ──────────────────────────────────────────────────────────────────────
    def _change_index(self):
        """Project the next pattern and schedule a single hardware trigger."""
        if not self.control_process:
            return

        self.index, self.index_max = self.projector.get_next_image()
        self.get_logger().info(f"Projecting image {self.index}/{self.index_max}")

        with self._timer_lock:
            # Cancel any leftover timer before creating a new one
            if self.trigger_timer is not None:
                self.trigger_timer.cancel()
                self.trigger_timer = None
            self.trigger_timer = self.create_timer(1.0 / self.hz, self._trigger)

    def _trigger(self):
        """Fire one hardware trigger then immediately cancel this timer."""
        with self._timer_lock:
            if self.trigger_timer is not None:
                self.trigger_timer.cancel()
                self.trigger_timer = None

        req = Trigger.Request()
        future = self.trigger_client.call_async(req)
        future.add_done_callback(self._trigger_callback)

    def _trigger_callback(self, future):
        try:
            response = future.result()
            if not response.success:
                self.get_logger().error('Hardware trigger returned failure.')
        except Exception as e:
            self.get_logger().error(f'Trigger service call error: {e}')

    # ──────────────────────────────────────────────────────────────────────
    # Publishing helpers
    # ──────────────────────────────────────────────────────────────────────
    def _publish_image(self, publisher, image: np.ndarray, encoding: str):
        assert isinstance(image, np.ndarray), f"Expected ndarray, got {type(image)}"
        try:
            msg = self.bridge.cv2_to_imgmsg(image, encoding=encoding)
            msg.header.stamp = self.get_clock().now().to_msg()
            publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f'Publish error: {e}')

    def _publish_debug_image(self, publisher, image_f64: np.ndarray,
                              frame_id: str = 'camera'):
        img_min = np.nanmin(image_f64)
        img_max = np.nanmax(image_f64)
        if img_max - img_min > 0:
            vis = ((image_f64 - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            vis = np.zeros_like(image_f64, dtype=np.uint8)
        msg = self.bridge.cv2_to_imgmsg(vis, encoding='mono8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        publisher.publish(msg)

    # ──────────────────────────────────────────────────────────────────────
    # Debug save
    # ──────────────────────────────────────────────────────────────────────
    def _debug_save_images(self, left: np.ndarray, right: np.ndarray, index: int):
        try:
            debug_dir = f'./{time.strftime("%Y%m%d")}_fringe_images'
            os.makedirs(os.path.join(debug_dir, 'left'),  exist_ok=True)
            os.makedirs(os.path.join(debug_dir, 'right'), exist_ok=True)
            left_path  = os.path.join(debug_dir, f'left/L{index:03d}.png')
            right_path = os.path.join(debug_dir, f'right/R{index:03d}.png')
            cv2.imwrite(left_path,  left)
            cv2.imwrite(right_path, right)
            self.get_logger().debug(f'Saved debug: {left_path} | {right_path}')
        except Exception as e:
            self.get_logger().error(f'Debug save error: {e}')


# ─────────────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = ProcessPhase()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()