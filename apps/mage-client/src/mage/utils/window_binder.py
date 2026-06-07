# MAGE — Gaming HUD for real-time screen translation.
# Copyright (C) 2026  Clementine Pendragon <clem@pendragon.systems>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Contact: clem@pendragon.systems (Clementine Pendragon, c/o Xian Project Development)

"""Cross-platform window binding utility.

Tracks a target window by title, retrieves its geometry, checks active status,
and handles graceful fallback on non-supported platforms (like Wayland or macOS).
"""

import logging
import sys
import ctypes
from ctypes import c_int, c_long, c_ulong, c_void_p, c_char_p, POINTER, Structure, byref

logger = logging.getLogger(__name__)

# --- X11 library loading and structures ---
X11 = None
if sys.platform.startswith("linux"):
    try:
        X11 = ctypes.CDLL("libX11.so.6")
    except OSError:
        try:
            X11 = ctypes.CDLL("libX11.so")
        except OSError:
            logger.warning("X11 client library (libX11) could not be loaded. X11 features disabled.")

class XWindowAttributes(Structure):
    _fields_ = [
        ("x", c_int),
        ("y", c_int),
        ("width", c_int),
        ("height", c_int),
        ("border_width", c_int),
        ("depth", c_int),
        ("visual", c_void_p),
        ("root", c_ulong),
        ("class_", c_int),
        ("bit_gravity", c_int),
        ("win_gravity", c_int),
        ("backing_store", c_int),
        ("backing_planes", c_ulong),
        ("backing_pixel", c_ulong),
        ("save_under", c_int),
        ("colormap", c_ulong),
        ("map_installed", c_int),
        ("map_state", c_int),  # 0 = IsUnmapped, 1 = IsUnviewable, 2 = IsViewable
        ("all_event_masks", c_ulong),
        ("your_event_mask", c_ulong),
        ("do_not_propagate_mask", c_ulong),
        ("override_redirect", c_int),
        ("screen", c_void_p),
    ]

# Setup X11 function signatures if available
if X11:
    try:
        X11.XOpenDisplay.argtypes = [c_char_p]
        X11.XOpenDisplay.restype = c_void_p
        
        X11.XCloseDisplay.argtypes = [c_void_p]
        X11.XCloseDisplay.restype = c_int
        
        X11.XDefaultRootWindow.argtypes = [c_void_p]
        X11.XDefaultRootWindow.restype = c_ulong
        
        X11.XQueryTree.argtypes = [
            c_void_p,          # Display*
            c_ulong,           # Window
            POINTER(c_ulong),  # Window* root_return
            POINTER(c_ulong),  # Window* parent_return
            POINTER(POINTER(c_ulong)), # Window** children_return
            POINTER(c_int)     # unsigned int* nchildren_return
        ]
        X11.XQueryTree.restype = c_int
        
        X11.XFree.argtypes = [c_void_p]
        X11.XFree.restype = c_int
        
        X11.XGetWindowAttributes.argtypes = [c_void_p, c_ulong, POINTER(XWindowAttributes)]
        X11.XGetWindowAttributes.restype = c_int
        
        X11.XFetchName.argtypes = [c_void_p, c_ulong, POINTER(c_char_p)]
        X11.XFetchName.restype = c_int
        
        X11.XInternAtom.argtypes = [c_void_p, c_char_p, c_int]
        X11.XInternAtom.restype = c_ulong
        
        X11.XGetWindowProperty.argtypes = [
            c_void_p,          # Display*
            c_ulong,           # Window
            c_ulong,           # Atom property
            c_long,            # long long_offset
            c_long,            # long long_length
            c_int,             # Bool delete
            c_ulong,           # Atom req_type
            POINTER(c_ulong),  # Atom* actual_type_return
            POINTER(c_int),    # int* actual_format_return
            POINTER(c_ulong),  # unsigned long* nitems_return
            POINTER(c_ulong),  # unsigned long* bytes_after_return
            POINTER(POINTER(ctypes.c_ubyte)) # unsigned char** prop_return
        ]
        X11.XGetWindowProperty.restype = c_int
        
        X11.XTranslateCoordinates.argtypes = [
            c_void_p,          # Display*
            c_ulong,           # Window src_w
            c_ulong,           # Window dest_w
            c_int,             # int src_x
            c_int,             # int src_y
            POINTER(c_int),    # int* dest_x_return
            POINTER(c_int),    # int* dest_y_return
            POINTER(c_ulong)   # Window* child_return
        ]
        X11.XTranslateCoordinates.restype = c_int
    except AttributeError as e:
        logger.error("Failed to map X11 ctypes: %s", e)
        X11 = None


# --- Windows structures ---
class RECT(Structure):
    _fields_ = [
        ("left", c_int),
        ("top", c_int),
        ("right", c_int),
        ("bottom", c_int),
    ]


# --- Helper functions for X11 ---
def get_window_title_x11(display, win) -> str | None:
    if not X11:
        return None
    name_ptr = c_char_p()
    if X11.XFetchName(display, win, byref(name_ptr)) != 0 and name_ptr.value:
        try:
            return name_ptr.value.decode('utf-8', errors='ignore')
        finally:
            X11.XFree(name_ptr)
            
    # Fallback to _NET_WM_NAME
    net_wm_name_atom = X11.XInternAtom(display, b"_NET_WM_NAME", False)
    utf8_string_atom = X11.XInternAtom(display, b"UTF8_STRING", False)
    
    actual_type_return = c_ulong()
    actual_format_return = c_int()
    nitems_return = c_ulong()
    bytes_after_return = c_ulong()
    prop_return = POINTER(ctypes.c_ubyte)()
    
    status = X11.XGetWindowProperty(
        display, win, net_wm_name_atom, 0, 1024, False, utf8_string_atom,
        byref(actual_type_return), byref(actual_format_return),
        byref(nitems_return), byref(bytes_after_return), byref(prop_return)
    )
    if status == 0 and prop_return:
        try:
            if nitems_return.value > 0:
                title_bytes = ctypes.string_at(prop_return, nitems_return.value)
                return title_bytes.decode('utf-8', errors='ignore')
        finally:
            X11.XFree(prop_return)
            
    return None

def find_window_x11(display, root_window, title_substring) -> int | None:
    if not X11:
        return None
    queue = [root_window]
    visited = set()
    
    while queue:
        win = queue.pop(0)
        if win in visited:
            continue
        visited.add(win)
        
        name = get_window_title_x11(display, win)
        if name and title_substring.lower() in name.lower():
            # Check map state to verify it's viewable (mapped on screen)
            attrs = XWindowAttributes()
            if X11.XGetWindowAttributes(display, win, byref(attrs)) != 0:
                if attrs.map_state == 2:  # IsViewable
                    return win
        
        # Query children
        root_return = c_ulong()
        parent_return = c_ulong()
        children_return = POINTER(c_ulong)()
        nchildren_return = c_int()
        
        status = X11.XQueryTree(display, win, byref(root_return), byref(parent_return), byref(children_return), byref(nchildren_return))
        if status != 0 and children_return:
            try:
                for i in range(nchildren_return.value):
                    child = children_return[i]
                    if child not in visited:
                        queue.append(child)
            finally:
                X11.XFree(children_return)
    return None

def get_window_geometry_x11(display, win) -> tuple[int, int, int, int] | None:
    if not X11:
        return None
    attrs = XWindowAttributes()
    if X11.XGetWindowAttributes(display, win, byref(attrs)) == 0:
        return None
        
    dest_x = c_int()
    dest_y = c_int()
    child = c_ulong()
    
    # Translate relative to root screen coordinates
    if X11.XTranslateCoordinates(display, win, attrs.root, 0, 0, byref(dest_x), byref(dest_y), byref(child)) != 0:
        return (dest_x.value, dest_y.value, attrs.width, attrs.height)
        
    return (attrs.x, attrs.y, attrs.width, attrs.height)

def is_window_active_x11(display, win) -> bool:
    if not X11:
        return False
    attrs = XWindowAttributes()
    if X11.XGetWindowAttributes(display, win, byref(attrs)) == 0:
        return False
        
    active_atom = X11.XInternAtom(display, b"_NET_ACTIVE_WINDOW", False)
    window_atom = X11.XInternAtom(display, b"WINDOW", False)
    
    actual_type_return = c_ulong()
    actual_format_return = c_int()
    nitems_return = c_ulong()
    bytes_after_return = c_ulong()
    prop_return = POINTER(ctypes.c_ubyte)()
    
    status = X11.XGetWindowProperty(
        display, attrs.root, active_atom, 0, 1024, False, window_atom,
        byref(actual_type_return), byref(actual_format_return),
        byref(nitems_return), byref(bytes_after_return), byref(prop_return)
    )
    if status == 0 and prop_return:
        try:
            if nitems_return.value > 0:
                active_win = POINTER(c_ulong)(prop_return)[0]
                return active_win == win
        finally:
            X11.XFree(prop_return)
    return False


def get_active_window_titles_x11(display) -> list[str]:
    if not X11 or not display:
        return []
    root = X11.XDefaultRootWindow(display)
    
    root_return = c_ulong()
    parent_return = c_ulong()
    children_return = POINTER(c_ulong)()
    nchildren_return = c_int()
    
    status = X11.XQueryTree(display, root, byref(root_return), byref(parent_return), byref(children_return), byref(nchildren_return))
    if status == 0 or not children_return:
        return []
        
    titles = []
    try:
        for i in range(nchildren_return.value):
            child = children_return[i]
            
            # BFS traverse to find the title
            sub_queue = [child]
            visited = set()
            found_title = None
            while sub_queue and len(visited) < 10:
                win = sub_queue.pop(0)
                if win in visited:
                    continue
                visited.add(win)
                
                attrs = XWindowAttributes()
                if X11.XGetWindowAttributes(display, win, byref(attrs)) != 0:
                    if attrs.map_state == 2:  # IsViewable
                        name = get_window_title_x11(display, win)
                        if name and name.strip():
                            found_title = name.strip()
                            break
                        
                        # Query children
                        sub_root = c_ulong()
                        sub_parent = c_ulong()
                        sub_children = POINTER(c_ulong)()
                        sub_nchildren = c_int()
                        if X11.XQueryTree(display, win, byref(sub_root), byref(sub_parent), byref(sub_children), byref(sub_nchildren)) != 0 and sub_children:
                            try:
                                for j in range(sub_nchildren.value):
                                    sub_queue.append(sub_children[j])
                            finally:
                                X11.XFree(sub_children)
            if found_title and found_title not in titles:
                titles.append(found_title)
    finally:
        X11.XFree(children_return)
    return sorted(titles)


# --- Helper functions for Windows ---
def find_window_windows(title_substring) -> int | None:
    found_hwnd = [None]
    
    # Define enum windows callback type
    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    
    def enum_cb(hwnd, lparam):
        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        if length > 0:
            buffer = ctypes.create_unicode_buffer(length + 1)
            ctypes.windll.user32.GetWindowTextW(hwnd, buffer, length + 1)
            if title_substring.lower() in buffer.value.lower():
                if ctypes.windll.user32.IsWindowVisible(hwnd):
                    found_hwnd[0] = hwnd
                    return False  # Stop enumeration
        return True
        
    callback = WNDENUMPROC(enum_cb)
    ctypes.windll.user32.EnumWindows(callback, 0)
    return found_hwnd[0]

def get_window_geometry_windows(hwnd) -> tuple[int, int, int, int] | None:
    rect = RECT()
    if ctypes.windll.user32.GetWindowRect(hwnd, byref(rect)):
        return (rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top)
    return None

def is_window_active_windows(hwnd) -> bool:
    return ctypes.windll.user32.GetForegroundWindow() == hwnd

def is_window_minimized_windows(hwnd) -> bool:
    return bool(ctypes.windll.user32.IsIconic(hwnd))


def get_active_window_titles_windows() -> list[str]:
    titles = []
    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    
    def enum_cb(hwnd, lparam):
        if ctypes.windll.user32.IsWindowVisible(hwnd):
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buffer = ctypes.create_unicode_buffer(length + 1)
                ctypes.windll.user32.GetWindowTextW(hwnd, buffer, length + 1)
                title = buffer.value.strip()
                if title and title not in titles:
                    titles.append(title)
        return True
        
    callback = WNDENUMPROC(enum_cb)
    ctypes.windll.user32.EnumWindows(callback, 0)
    return sorted(titles)


# --- Main Wrapper Class ---
class WindowBinder:
    def __init__(self, target_title: str):
        self.target_title = target_title.strip()
        self.platform = None
        self._win_id = None
        self._x11_display = None
        
        # Detect platform and environment
        if sys.platform == "win32":
            self.platform = "windows"
        elif sys.platform == "darwin":
            self.platform = "macos"
        else:
            # Check Qt platform name if QApplication is running, otherwise check environment variables
            from PyQt6.QtGui import QGuiApplication
            qt_platform = QGuiApplication.platformName() if QGuiApplication.instance() else None
            
            if qt_platform == "xcb" or (not qt_platform and "DISPLAY" in os_environ_check()):
                self.platform = "x11"
            elif qt_platform == "wayland":
                self.platform = "wayland"
                
        # Initialize Display for X11 if needed
        if self.platform == "x11" and X11:
            try:
                self._x11_display = X11.XOpenDisplay(None)
            except Exception as e:
                logger.error("Failed to open X11 display: %s", e)
                self.platform = "x11_failed"
                
    def close(self):
        if self._x11_display and X11:
            try:
                X11.XCloseDisplay(self._x11_display)
            except Exception as e:
                logger.error("Error closing X11 display: %s", e)
            self._x11_display = None

    def _is_cached_window_valid(self) -> bool:
        if not self._win_id:
            return False
        try:
            if self.platform == "windows":
                user32 = ctypes.windll.user32
                if not user32.IsWindow(self._win_id):
                    return False
                length = user32.GetWindowTextLengthW(self._win_id)
                if length == 0:
                    return False
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(self._win_id, buf, length + 1)
                return self.target_title.lower() in buf.value.lower()
            elif self.platform == "x11" and self._x11_display and X11:
                attrs = XWindowAttributes()
                if X11.XGetWindowAttributes(self._x11_display, self._win_id, byref(attrs)) == 0:
                    return False
                name = get_window_title_x11(self._x11_display, self._win_id)
                return name is not None and self.target_title.lower() in name.lower()
        except Exception as e:
            logger.debug("Error verifying cached window validity: %s", e)
        return False

    def update_target(self):
        """Locates the window ID/handle matching the target title."""
        if not self.target_title:
            self._win_id = None
            return
            
        if self._win_id is not None and self._is_cached_window_valid():
            return

        try:
            if self.platform == "windows":
                self._win_id = find_window_windows(self.target_title)
            elif self.platform == "x11" and self._x11_display and X11:
                root = X11.XDefaultRootWindow(self._x11_display)
                self._win_id = find_window_x11(self._x11_display, root, self.target_title)
            else:
                self._win_id = None
        except Exception as e:
            logger.error("Error resolving target window: %s", e)
            self._win_id = None

    def exists(self) -> bool:
        self.update_target()
        return self._win_id is not None
        
    def get_geometry(self) -> tuple[int, int, int, int] | None:
        """Returns (x, y, width, height) of the target window in screen coordinates."""
        if not self._win_id:
            self.update_target()
        if not self._win_id:
            return None
            
        try:
            if self.platform == "windows":
                return get_window_geometry_windows(self._win_id)
            elif self.platform == "x11" and self._x11_display and X11:
                return get_window_geometry_x11(self._x11_display, self._win_id)
        except Exception as e:
            logger.error("Error fetching target geometry: %s", e)
        return None

    def is_active(self) -> bool:
        """Returns True if the target window is the foreground/active window."""
        if not self._win_id:
            self.update_target()
        if not self._win_id:
            return False
            
        try:
            if self.platform == "windows":
                return is_window_active_windows(self._win_id)
            elif self.platform == "x11" and self._x11_display and X11:
                return is_window_active_x11(self._x11_display, self._win_id)
        except Exception as e:
            logger.error("Error checking target active status: %s", e)
        return False

    def is_minimized(self) -> bool:
        if not self._win_id:
            self.update_target()
        if not self._win_id:
            return False
            
        try:
            if self.platform == "windows":
                return is_window_minimized_windows(self._win_id)
            elif self.platform == "x11" and self._x11_display and X11:
                attrs = XWindowAttributes()
                if X11.XGetWindowAttributes(self._x11_display, self._win_id, byref(attrs)) != 0:
                    return attrs.map_state != 2
        except Exception as e:
            logger.error("Error checking target minimized status: %s", e)
        return False

    def get_native_id(self) -> int | None:
        """Returns the native window handle/ID."""
        return self._win_id

    @classmethod
    def get_active_window_titles(cls) -> list[str]:
        import sys
        platform = None
        if sys.platform == "win32":
            platform = "windows"
        elif sys.platform == "darwin":
            platform = "macos"
        else:
            from PyQt6.QtGui import QGuiApplication
            qt_platform = QGuiApplication.platformName() if QGuiApplication.instance() else None
            
            if qt_platform == "xcb" or (not qt_platform and "DISPLAY" in os_environ_check()):
                platform = "x11"
                
        if platform == "windows":
            return get_active_window_titles_windows()
        elif platform == "x11" and X11:
            try:
                display = X11.XOpenDisplay(None)
                if display:
                    try:
                        return get_active_window_titles_x11(display)
                    finally:
                        X11.XCloseDisplay(display)
            except Exception as e:
                logger.error("Error listing X11 window titles: %s", e)
        return []


def os_environ_check():
    import os
    return os.environ


_bypass_hint_display = None


def set_bypass_compositor_hint_x11(win_id):
    """Set _NET_WM_BYPASS_COMPOSITOR to 2 (don't bypass) to keep compositor active on Linux."""
    global _bypass_hint_display
    if not sys.platform.startswith("linux"):
        return
    if not X11:
        return
    
    # Do not execute X11 calls if running natively under Wayland
    from PyQt6.QtGui import QGuiApplication
    qt_platform = QGuiApplication.platformName() if QGuiApplication.instance() else None
    if qt_platform != "xcb":
        return
        
    try:
        if win_id is not None:
            try:
                win_id = int(win_id)
            except (TypeError, ValueError):
                pass
        if _bypass_hint_display is None:
            _bypass_hint_display = X11.XOpenDisplay(None)
        display = _bypass_hint_display
        if not display:
            return
        
        atom = X11.XInternAtom(display, b"_NET_WM_BYPASS_COMPOSITOR", False)
        cardinal_atom = X11.XInternAtom(display, b"CARDINAL", False)
        
        # 2 = Don't bypass compositor (forces composition to stay enabled)
        data = (c_ulong * 1)(2)
        
        # Define argtypes dynamically
        X11.XChangeProperty.argtypes = [
            c_void_p, c_ulong, c_ulong, c_ulong, c_int, c_int, POINTER(c_ulong), c_int
        ]
        X11.XChangeProperty.restype = c_int
        
        if hasattr(X11, "XFlush"):
            X11.XFlush.argtypes = [c_void_p]
            X11.XFlush.restype = c_int
            
        X11.XChangeProperty(
            display,
            win_id,
            atom,
            cardinal_atom,
            32,
            0, # PropModeReplace
            data,
            1
        )
        if hasattr(X11, "XFlush"):
            X11.XFlush(display)
        logger.info("Set _NET_WM_BYPASS_COMPOSITOR to 2 on window XID: %s", win_id)
    except Exception as e:
        logger.debug("Failed to set _NET_WM_BYPASS_COMPOSITOR: %s", e)
