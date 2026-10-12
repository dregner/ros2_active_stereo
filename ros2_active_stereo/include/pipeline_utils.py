"""ROS-independent helpers for acquisition ownership and dataset validation."""
import datetime
import math
import os
from pathlib import Path
import uuid
import json
import threading
import time
import gc
from concurrent.futures import ThreadPoolExecutor


def unique_scan_folder(save_images):
    base = Path.home() / ('Pictures/active_stereo' if save_images else 'active_results')
    name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    return str(base / f'{name}_{uuid.uuid4().hex[:8]}')


def graycode_bits(width, pixels_per_fringe):
    if width <= 0 or pixels_per_fringe <= 0:
        raise ValueError('Pattern dimensions and period must be positive')
    # Keep identical to GrayCode.cpp; do not floor a fractional fringe count.
    return math.ceil(math.log2(width / pixels_per_fringe) + 1.0)


def completion_token(line):
    fields = line.strip().split()
    if len(fields) >= 2 and fields[0] == '[Pipeline]':
        if fields[1] in {'FRINGE_DONE', 'CORRELATION_DONE', 'MOTOR_DONE', 'LASER_DONE'}:
            return fields[1]
    return None


def validate_pairs(folder, kind, count):
    """Require exact indexed files; reject leftovers and incomplete stereo sets."""
    left = Path(folder) / kind / 'left'
    right = Path(folder) / kind / 'right'
    expected_l = [f'L{i:03d}.png' for i in range(count)]
    expected_r = [f'R{i:03d}.png' for i in range(count)]
    if sorted(os.listdir(left)) != expected_l or sorted(os.listdir(right)) != expected_r:
        raise ValueError(f'{kind}: expected exactly {count} indexed stereo pairs')
    return [(str(left / l), str(right / r)) for l, r in zip(expected_l, expected_r)]


def stamp_key(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


class ScanWorkerMixin:
    """One acquisition/reconstruction job per node, with explicit ownership."""

    def _init_scan_worker(self, kind, busy_field):
        from std_msgs.msg import String
        self._scan_kind = kind
        self._busy_field = busy_field
        self._scan_lock = threading.RLock()
        self._worker = ThreadPoolExecutor(max_workers=1)
        self._working = False
        self._generation = 0
        self._scan_stamp = None
        self._expected_stamp = None
        self.declare_parameter('acquisition_timeout_s', 180.0)
        self._result_sub = self.create_subscription(
            String, 'acquisition_result', self._result_callback, 10)
        self._watchdog = self.create_timer(0.5, self._scan_watchdog)

    def _begin_scan(self, client, request_type):
        with self._scan_lock:
            if getattr(self, self._busy_field):
                self.get_logger().warning('Processing already active; request rejected')
                return
            if not client.service_is_ready():
                self.get_logger().error('Acquisition service is unavailable')
                return
            self._reset_scan_buffers()
            self.folder = unique_scan_folder(bool(self.get_parameter('save_images').value))
            self._generation += 1
            generation = self._generation
            setattr(self, self._busy_field, True)
            self._working = False
            self._expected_stamp = None
            self._scan_stamp = None
            self._deadline = time.monotonic() + float(self.get_parameter('acquisition_timeout_s').value)
            request = request_type.Request()
            request.folder_path = self.folder
            try:
                future = client.call_async(request)
                future.add_done_callback(lambda f: self._request_finished(f, generation))
            except Exception as exc:
                self.get_logger().error(f'Acquisition request failed: {exc}')
                self._finish_scan()

    def _request_finished(self, future, generation):
        with self._scan_lock:
            if generation != self._generation or not getattr(self, self._busy_field):
                return
            try:
                result = future.result()
                if result is None or not result.success:
                    raise RuntimeError('No response' if result is None else result.message)
            except Exception as exc:
                self.get_logger().error(f'Acquisition rejected: {exc}')
                if not self._working:
                    self._finish_scan()

    def _result_callback(self, msg):
        try:
            event = json.loads(msg.data)
            if not isinstance(event, dict):
                raise ValueError('Result must be an object')
        except (TypeError, ValueError):
            self.get_logger().warning('Ignoring malformed acquisition result')
            return
        with self._scan_lock:
            if (not getattr(self, self._busy_field) or self._working or
                    event.get('kind') != self._scan_kind or event.get('folder') != self.folder):
                return
            if event.get('success') is not True:
                self.get_logger().error(f"Acquisition failed: {event.get('message', '')}")
                self._finish_scan()
                return
            stamp = event.get('stamp')
            if stamp is not None:
                from builtin_interfaces.msg import Time
                if not isinstance(stamp, list) or len(stamp) != 2:
                    self.get_logger().error('Invalid acquisition timestamp')
                    self._finish_scan()
                    return
                if (not all(isinstance(x, int) for x in stamp) or
                        not 0 <= stamp[1] < 1_000_000_000):
                    self.get_logger().error('Invalid acquisition timestamp')
                    self._finish_scan()
                    return
                self._scan_stamp = Time(sec=stamp[0], nanosec=stamp[1])
                self._expected_stamp = stamp_key(self._scan_stamp)
            self._dataset_ready()

    def _submit_scan(self, callback):
        if self._working:
            return
        self._working = True
        self._worker.submit(self._process_scan, callback)

    def _process_scan(self, callback):
        try:
            callback()
        except Exception as exc:
            self.get_logger().error(f'Reconstruction failed: {exc}')
        finally:
            with self._scan_lock:
                self._finish_scan()

    def _finish_scan(self):
        # Release live tensors first; empty_cache alone cannot release them.
        for name in ('left_images', 'right_images', 'grid', 'x_vals', 'y_vals', 'z_vals'):
            setattr(self.zscan, name, None)
        self._reset_scan_buffers()
        self._expected_stamp = None
        self._working = False
        setattr(self, self._busy_field, False)
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _scan_watchdog(self):
        with self._scan_lock:
            if (getattr(self, self._busy_field) and not self._working and
                    time.monotonic() > self._deadline):
                self.get_logger().error('Timed out waiting for the complete acquisition dataset')
                self._finish_scan()

    def destroy_node(self):
        worker = getattr(self, '_worker', None)
        if worker is not None:
            worker.shutdown(wait=True)
        return super().destroy_node()
