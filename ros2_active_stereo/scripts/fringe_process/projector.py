import cv2
import numpy as np
import screeninfo


class ProjectorControl:
    def __init__(self, img_resolution=(1920, 1080), monitor_name='DP-1', index=0):
        self.img_resolution = img_resolution
        self.current_image_index = index
        self.fringe_images = []
        self.graycode_images = []

        # Detect monitor and configure position
        self.move = self._detect_monitor(monitor_name)

        # Combined patterns
        self.n_img = None
        self.n_img_max = 0

    def set_images(self, fringe, graycode):
        """Define fringe and Gray Code pattern images."""
        self.fringe_images = fringe
        self.graycode_images = graycode
        self.n_img = self._combine_patterns()
        self.n_img_max = self.n_img.shape[2] - 1

    def _detect_monitor(self, monitor_name):
        """Detects monitor by name or falls back to secondary/primary monitor."""
        monitors = screeninfo.get_monitors()
        for m in monitors:
            if monitor_name in m.name:
                self.img_resolution = (m.width, m.height)
                return (m.x, m.y)
        # Fallback to second monitor if available, else primary
        if len(monitors) > 1:
            m = monitors[1]
            self.img_resolution = (m.width, m.height)
            return (m.x, m.y)
        elif len(monitors) == 1:
            m = monitors[0]
            self.img_resolution = (m.width, m.height)
            return (m.x, m.y)
        return (0, 0)

    def _combine_patterns(self):
        """Combines Gray Code and Fringe pattern images along channels."""
        return np.concatenate((self.graycode_images, self.fringe_images), axis=2)

    def setup_projector_window(self):
        """Creates fullscreen projection window on target monitor."""
        cv2.namedWindow('projector', cv2.WINDOW_NORMAL)
        cv2.moveWindow('projector', self.move[0], self.move[1])
        cv2.setWindowProperty('projector', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        black = np.zeros((self.img_resolution[1], self.img_resolution[0], 3), dtype=np.uint8)
        self.project_image(black)

    def get_next_image(self):
        """Advances and projects the next pattern image."""
        self.current_image_index = (self.current_image_index + 1) % (self.n_img_max + 1)
        pattern_image = self.n_img[:, :, self.current_image_index]
        pattern_image = self.convert_to_BGR(pattern_image)

        cv2.imshow('projector', pattern_image)
        cv2.waitKey(1)
        return self.current_image_index, self.n_img_max

    def project_image(self, image):
        """Projects a specific image (e.g. black background)."""
        cv2.imshow('projector', image)
        cv2.waitKey(1)

    def convert_to_BGR(self, image):
        """Converts monochrome pattern to blue channel for laser projection."""
        bgr_image = np.zeros((*image.shape, 3), dtype=np.uint8)
        bgr_image[:, :, 0] = image  # Blue channel
        return bgr_image
