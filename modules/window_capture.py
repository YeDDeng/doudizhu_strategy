"""
Module 1: Window Capture
Captures Doudizhu game window using win32gui and mss.
Supports background capture at ≥10 FPS.
"""

import win32gui
import mss
import numpy as np
from typing import Tuple, Optional, Dict, List

# Lazy cv2 import to avoid cv2 recursion issue with Python 3.14
_cv2 = None
def _get_cv2():
    global _cv2
    if _cv2 is None:
        import cv2 as _cv2_module
        _cv2 = _cv2_module
    return _cv2

class WindowCapture:
    """Game window capture class."""

    def __init__(self, window_title_keywords: List[str] = None):
        """Initialize with window title keywords to search."""
        self.window_title_keywords = window_title_keywords or [
            "欢乐斗地主", "腾讯斗地主", "JJ斗地主", "斗地主"
        ]
        self.hwnd: Optional[int] = None
        self._last_rect: Optional[Tuple[int, int, int, int]] = None
        self._custom_rect: Optional[Tuple[int, int, int, int]] = None
        # Smooth rect over last N captures to reduce jitter
        self._rect_history: List[Tuple[int, int, int, int]] = []
        self._rect_history_maxlen = 5

    def find_window(self) -> Optional[int]:
        """Find the game window by title keywords."""
        def callback(hwnd, extra):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                for keyword in self.window_title_keywords:
                    if keyword in title:
                        extra.append((hwnd, title))
            return True

        hwnds = []
        win32gui.EnumWindows(callback, hwnds)

        if not hwnds:
            return None

        # Filter and find the largest window (most likely the actual game window)
        valid_windows = []
        for hwnd, title in hwnds:
            try:
                rect = win32gui.GetWindowRect(hwnd)
                width = rect[2] - rect[0]
                height = rect[3] - rect[1]
                if width >= 800 and height >= 600:  # Reasonable game window size
                    valid_windows.append((width * height, hwnd, title))
            except:
                continue

        if not valid_windows:
            # If no large window, take the first one anyway
            print(f"Found {len(hwnds)} windows matching title, none large enough. Taking first: '{hwnds[0][1]}'")
            self.hwnd = hwnds[0][0]
            return self.hwnd

        # Sort by area (largest first)
        valid_windows.sort(reverse=True, key=lambda x: x[0])
        area, hwnd, title = valid_windows[0]
        print(f"Found {len(valid_windows)} valid game windows. Selected largest: '{title}' {area}px")
        self.hwnd = hwnd
        return self.hwnd

        return None

    def find_window_on_screen(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Find the Doudizhu game window by visual detection on screen.
        Uses color detection to find the green game table.

        Returns:
            Window rect (left, top, right, bottom) if found, None otherwise.
        """
        try:
            with mss.mss() as sct:
                # Capture full screen
                screen = sct.grab(sct.monitors[1])
                img = np.array(screen)
                if img.shape[2] == 4:
                    img = img[:, :, :3]

                # Convert to HSV for better color detection
                hsv = _get_cv2().cvtColor(img, _get_cv2().COLOR_BGR2HSV)

                # Green color range for game table (typical Doudizhu green felt)
                lower_green = np.array([35, 30, 40])
                upper_green = np.array([85, 100, 150])

                mask = _get_cv2().inRange(hsv, lower_green, upper_green)

                # Find contours
                contours, _ = _get_cv2().findContours(mask, _get_cv2().RETR_EXTERNAL, _get_cv2().CHAIN_APPROX_SIMPLE)

                if not contours:
                    print("[find_window_on_screen] No green area found on screen")
                    return None

                # Find the largest contour (likely the game table)
                largest = max(contours, key=_get_cv2().contourArea)
                area = _get_cv2().contourArea(largest)

                # Filter by minimum area (avoid tiny green spots)
                min_area = 50000  # ~500x100 pixels minimum
                if area < min_area:
                    print(f"[find_window_on_screen] Largest green area too small: {area}")
                    return None

                # Get bounding rect
                x, y, w, h = _get_cv2().boundingRect(largest)

                # Add some padding
                pad = 5
                left = max(0, x - pad)
                top = max(0, y - pad)
                right = min(img.shape[1], x + w + pad)
                bottom = min(img.shape[0], y + h + pad)

                print(f"[find_window_on_screen] Found game area: ({left},{top},{right},{bottom}) size={right-left}x{bottom-top}")
                return (left, top, right, bottom)

        except Exception as e:
            print(f"[find_window_on_screen] Error: {e}")
            return None

    def set_capture_region(self, rect: Tuple[int, int, int, int]) -> None:
        """
        Set a custom capture region instead of using window handle.
        Useful when window detection by title doesn't work.

        Args:
            rect: (left, top, right, bottom) coordinates
        """
        self._custom_rect = rect
        self.hwnd = None  # Disable window-based capture
        print(f"[set_capture_region] Set capture region: {rect}")

    def get_window_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """Get window rectangle (left, top, right, bottom), smoothed over last N captures."""
        # Use custom rect if set
        if self._custom_rect is not None:
            self._last_rect = self._custom_rect
            return self._custom_rect

        if self.hwnd is None:
            return None

        try:
            rect = win32gui.GetWindowRect(self.hwnd)
            left, top, right, bottom = rect
            width = right - left
            height = bottom - top
            if width <= 0 or height <= 0:
                print(f"Window has invalid size: {width}x{height}, hwnd={self.hwnd}")
                return None

            # Add to history and compute moving average for smoothing
            self._rect_history.append(rect)
            if len(self._rect_history) > self._rect_history_maxlen:
                self._rect_history.pop(0)

            # Average all values in history for stable rect
            n = len(self._rect_history)
            avg_l = sum(r[0] for r in self._rect_history) // n
            avg_t = sum(r[1] for r in self._rect_history) // n
            avg_r = sum(r[2] for r in self._rect_history) // n
            avg_b = sum(r[3] for r in self._rect_history) // n
            smoothed = (avg_l, avg_t, avg_r, avg_b)

            self._last_rect = smoothed
            return smoothed
        except Exception as e:
            print(f"Failed to get window rect: {e}, hwnd={self.hwnd}")
            return None

    def capture(self) -> Optional[np.ndarray]:
        """Capture the window as a numpy array (BGR format for OpenCV)."""
        rect = self.get_window_rect()
        if rect is None:
            return None

        left, top, right, bottom = rect
        width = right - left
        height = bottom - top

        if width <= 0 or height <= 0:
            return None

        monitor = {"top": top, "left": left, "width": width, "height": height}

        try:
            # Create mss instance in current thread to fix '_thread._local' error
            with mss.mss() as sct:
                img = sct.grab(monitor)
                # Convert to numpy array (BGRA -> BGR)
                img_array = np.array(img)
                if img_array.shape[2] == 4:
                    img_array = img_array[:, :, :3]
                return img_array
        except Exception as e:
            print(f"Failed to capture: {e}")
            return None

    def get_relative_regions(self, window_rect: Tuple[int, int, int, int]) -> Dict[str, Tuple[float, float, float, float]]:
        """
        Get relative regions for different game areas (0.0-1.0 coordinates).
        Adapts to window size dynamically.

        Returns:
            Dict with region name -> (left_rel, top_rel, right_rel, bottom_rel)
        """
        left, top, right, bottom = window_rect
        width = right - left
        height = bottom - top

        # Adaptive regions based on actual window size
        # These ratios should work for different resolutions
        return {
            "my_hand": (0.05, 0.60, 0.95, 0.95),      # My hand cards at bottom
            "upper_player_area": (0.10, 0.25, 0.50, 0.45),  # Upper opponent played cards
            "lower_player_area": (0.50, 0.25, 0.90, 0.45),  # Lower opponent played cards
            "center_played": (0.25, 0.20, 0.75, 0.50),      # Center played cards area
            "scorekeeper": (0.35, 0.02, 0.65, 0.15),   # Top scorekeeper area
            "upper_count": (0.02, 0.02, 0.15, 0.12),   # Upper player card count
            "lower_count": (0.85, 0.02, 0.98, 0.12),  # Lower player card count
            "landlord_mark": (0.40, 0.15, 0.60, 0.25), # Landlord indicator
            "turn_indicator": (0.35, 0.50, 0.65, 0.60), # Current turn indicator
        }

    def get_absolute_region(self, relative_region: Tuple[float, float, float, float],
                           window_rect: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Convert relative region to absolute pixels."""
        left, top, right, bottom = window_rect
        width = right - left
        height = bottom - top

        r_left, r_top, r_right, r_bottom = relative_region

        abs_left = left + int(r_left * width)
        abs_top = top + int(r_top * height)
        abs_right = left + int(r_right * width)
        abs_bottom = top + int(r_bottom * height)

        return (abs_left, abs_top, abs_right, abs_bottom)

    def capture_region(self, relative_region: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        """Capture a specific relative region of the window."""
        rect = self.get_window_rect()
        if rect is None:
            return None

        abs_region = self.get_absolute_region(relative_region, rect)
        left, top, right, bottom = abs_region
        width = right - left
        height = bottom - top

        if width <= 0 or height <= 0:
            return None

        monitor = {"top": top, "left": left, "width": width, "height": height}

        try:
            # Create mss instance in current thread to fix '_thread._local' error
            with mss.mss() as sct:
                img = sct.grab(monitor)
                img_array = np.array(img)
                if img_array.shape[2] == 4:
                    img_array = img_array[:, :, :3]
                return img_array
        except Exception as e:
            print(f"Failed to capture region: {e}")
            return None


if __name__ == "__main__":
    import time
    import cv2

    print("Testing WindowCapture...")

    capture = WindowCapture()
    hwnd = capture.find_window()

    if hwnd is None:
        print("No game window found. Please open a Doudizhu game first.")
        exit(1)

    print(f"Found window, hwnd: {hwnd}")

    rect = capture.get_window_rect()
    print(f"Window rect: {rect}")

    # Test capture FPS
    fps_test_count = 30
    start_time = time.time()

    for i in range(fps_test_count):
        img = capture.capture()
        if img is None:
            print("Capture failed")
            break

    end_time = time.time()
    fps = fps_test_count / (end_time - start_time)
    print(f"Average FPS over {fps_test_count} captures: {fps:.2f}")

    if fps >= 10:
        print("[OK] FPS meets requirement (≥ 10 FPS)")
    else:
        print("[FAIL] FPS below target")

    # Save a test capture
    test_img = capture.capture()
    if test_img is not None:
        _get_cv2().imwrite("../tests/capture_test.png", test_img)
        print("Test capture saved to tests/capture_test.png")

    _get_cv2().destroyAllWindows()
