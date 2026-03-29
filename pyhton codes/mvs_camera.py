"""Hikvision MVS SDK Python wrapper via ctypes.

Provides an OpenCV-compatible interface for MVS industrial cameras
(USB3 Vision, GigE Vision) using MvCameraControl.dll.

Usage:
    from src.mvs_camera import mvs_available, enumerate_mvs_cameras, MVSCamera

    if mvs_available():
        for idx, name in enumerate_mvs_cameras():
            print(f"MVS:{idx}  {name}")

        cam = MVSCamera()
        if cam.open(0):
            ok, frame = cam.read()   # frame is BGR numpy array
            cam.release()
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
import threading
import time
import numpy as np

# ── Optional cv2 for color conversion ────────────────────────────────────────
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# ── DLL search paths ──────────────────────────────────────────────────────────
_MVS_CANDIDATES = [
    r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64",
    r"C:\Program Files\Common Files\MVS\Runtime\Win64_x64",
    r"C:\Program Files (x86)\MVS\Runtime\Win64_x64",
    r"C:\Program Files\MVS\Runtime\Win64_x64",
]

_MVS_RUNTIME_DIR: str | None = None
_mvs_lib: ctypes.WinDLL | None = None   # type: ignore[type-arg]


def _runtime_candidates() -> list[str]:
    out: list[str] = []

    # Prefer bundled DLL folder when running from a PyInstaller EXE.
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", "")
        exe_dir = os.path.dirname(sys.executable)
        for d in (
            meipass,
            os.path.join(meipass, "_internal") if meipass else "",
            exe_dir,
            os.path.join(exe_dir, "_internal") if exe_dir else "",
        ):
            if d:
                out.append(d)

    # Allow override via environment variable.
    env_dir = os.getenv("MVS_RUNTIME_DIR", "").strip()
    if env_dir:
        out.append(env_dir)

    out.extend(_MVS_CANDIDATES)

    # Preserve order while removing duplicates.
    dedup: list[str] = []
    seen: set[str] = set()
    for path in out:
        norm = os.path.normcase(os.path.normpath(path))
        if norm in seen:
            continue
        seen.add(norm)
        dedup.append(path)
    return dedup


def _find_runtime_dir() -> str | None:
    for d in _runtime_candidates():
        if os.path.isfile(os.path.join(d, "MvCameraControl.dll")):
            return d
    return None


def _load_mvs() -> "ctypes.WinDLL | None":
    global _mvs_lib, _MVS_RUNTIME_DIR
    if _mvs_lib is not None:
        return _mvs_lib

    _MVS_RUNTIME_DIR = _find_runtime_dir()
    if _MVS_RUNTIME_DIR is None:
        return None

    # Allow Windows to find dependent DLLs in the same folder (Python 3.8+)
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(_MVS_RUNTIME_DIR)
        except Exception:
            pass
    try:
        os.environ["PATH"] = _MVS_RUNTIME_DIR + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

    dll_path = os.path.join(_MVS_RUNTIME_DIR, "MvCameraControl.dll")
    try:
        _mvs_lib = ctypes.WinDLL(dll_path)   # type: ignore[assignment]
        return _mvs_lib
    except OSError:
        return None


# ── Constants ─────────────────────────────────────────────────────────────────
MV_GIGE_DEVICE      = 0x00000001
MV_USB_DEVICE       = 0x00000004
MV_ACCESS_Exclusive = 1   # only one controller, no monitoring
MV_ACCESS_Control   = 2   # one controller, multiple monitors allowed
MV_ACCESS_Monitor   = 3   # read-only monitor (stream only)
MV_OK = 0

# GenICam enum values used by Hikvision cameras.
TRIGGER_MODE_OFF = 0
TRIGGER_MODE_ON = 1
TRIGGER_SOURCE_LINE0 = 0
TRIGGER_SOURCE_LINE1 = 1
TRIGGER_SOURCE_LINE2 = 2
TRIGGER_SOURCE_LINE3 = 3
TRIGGER_SOURCE_SOFTWARE = 7

# XML node access mode (CameraParams.h: MV_XML_AccessMode)
AM_NI = 0  # not implemented
AM_NA = 1  # not available
AM_WO = 2  # write only
AM_RO = 3  # read only
AM_RW = 4  # read/write

# Pixel type constants (GenICam PFNC)
PixelType_Gvsp_Mono8          = 0x01080001
PixelType_Gvsp_BayerGR8       = 0x01080008
PixelType_Gvsp_BayerRG8       = 0x01080009
PixelType_Gvsp_BayerGB8       = 0x0108000A
PixelType_Gvsp_BayerBG8       = 0x0108000B
PixelType_Gvsp_RGB8_Packed    = 0x02180014
PixelType_Gvsp_BGR8_Packed    = 0x02180015
PixelType_Gvsp_YUV422_Packed  = 0x02100032
PixelType_Gvsp_YUV422_YUYV    = 0x02100033

# ── ctypes structures ─────────────────────────────────────────────────────────
from ctypes import (
    c_ubyte, c_ushort, c_uint, c_uint64, c_void_p, POINTER, Union, Structure
)


class MV_GIGE_DEVICE_INFO(Structure):
    _fields_ = [
        ("nIpCfgOption",               c_uint),
        ("nIpCfgCurrent",              c_uint),
        ("nCurrentIp",                 c_uint),
        ("nCurrentSubNetMask",         c_uint),
        ("nDefultGateWay",             c_uint),
        ("chManufacturerName",         c_ubyte * 32),
        ("chModelName",                c_ubyte * 32),
        ("chDeviceVersion",            c_ubyte * 32),
        ("chManufacturerSpecificInfo", c_ubyte * 48),
        ("chSerialNumber",             c_ubyte * 16),
        ("chUserDefinedName",          c_ubyte * 16),
        ("nMacAddrHigh",               c_uint),
        ("nMacAddrLow",                c_uint),
        ("nNetExport",                 c_uint),
        ("nReserved",                  c_uint * 4),
    ]


class MV_USB3_DEVICE_INFO(Structure):
    _fields_ = [
        ("CrtlInEndPoint",    c_ubyte),
        ("CrtlOutEndPoint",   c_ubyte),
        ("StreamEndPoint",    c_ubyte),
        ("EventEndPoint",     c_ubyte),
        ("idVendor",          c_ushort),
        ("idProduct",         c_ushort),
        ("nDeviceNumber",     c_uint),
        ("chDeviceGUID",      c_ubyte * 64),
        ("chVendorName",      c_ubyte * 64),
        ("chModelName",       c_ubyte * 64),
        ("chFamilyName",      c_ubyte * 64),
        ("chDeviceVersion",   c_ubyte * 64),
        ("chManufacturerName", c_ubyte * 64),
        ("chSerialNumber",    c_ubyte * 64),
        ("chUserDefinedName", c_ubyte * 64),
        ("nbcdUSB",           c_uint),
        ("nReserved",         c_uint * 3),
    ]


class _SPECIAL_INFO(Union):
    _fields_ = [
        ("stGigEInfo",  MV_GIGE_DEVICE_INFO),
        ("stUsb3VInfo", MV_USB3_DEVICE_INFO),
    ]


class MV_CC_DEVICE_INFO(Structure):
    _fields_ = [
        ("nMajorVer",    c_ushort),
        ("nMinorVer",    c_ushort),
        ("nMacAddrHigh", c_uint),
        ("nMacAddrLow",  c_uint),
        ("nTLayerType",  c_uint),
        ("nReserved",    c_uint * 4),
        ("SpecialInfo",  _SPECIAL_INFO),
    ]


class MV_CC_DEVICE_INFO_LIST(Structure):
    _fields_ = [
        ("nDeviceNum",  c_uint),
        ("pDeviceInfo", POINTER(MV_CC_DEVICE_INFO) * 256),
    ]


class MV_FRAME_OUT_INFO_EX(Structure):
    _fields_ = [
        ("nWidth",            c_ushort),
        ("nHeight",           c_ushort),
        ("enPixelType",       c_uint),
        ("nFrameNum",         c_uint),
        ("nDevTimeStampHigh", c_uint),
        ("nDevTimeStampLow",  c_uint),
        ("nReserved0",        c_uint),
        ("nHostTimeStamp",    c_uint64),
        ("nFrameLen",         c_uint),
        ("nLostPacket",       c_uint),
        ("bReserved",         c_uint),
        ("nReserved",         c_uint * 25),
    ]


class MV_FRAME_OUT(Structure):
    _fields_ = [
        ("pBufAddr",    c_void_p),
        ("stFrameInfo", MV_FRAME_OUT_INFO_EX),
        ("nRes",        c_uint * 16),
    ]


# ── Public API ────────────────────────────────────────────────────────────────

def mvs_available() -> bool:
    """Return True if MvCameraControl.dll is loadable on this machine."""
    return _load_mvs() is not None


def enumerate_mvs_cameras() -> list[tuple[int, str]]:
    """Return list of (index, model_name) for all detected MVS cameras."""
    lib = _load_mvs()
    if lib is None:
        return []

    device_list = MV_CC_DEVICE_INFO_LIST()
    ret = lib.MV_CC_EnumDevices(
        MV_GIGE_DEVICE | MV_USB_DEVICE,
        ctypes.byref(device_list)
    )
    if ret != MV_OK:
        return []

    cameras: list[tuple[int, str]] = []
    for i in range(device_list.nDeviceNum):
        try:
            info = device_list.pDeviceInfo[i].contents
            if info.nTLayerType == MV_USB_DEVICE:
                raw = bytes(info.SpecialInfo.stUsb3VInfo.chModelName)
            else:
                raw = bytes(info.SpecialInfo.stGigEInfo.chModelName)
            name = raw.split(b"\x00")[0].decode("utf-8", errors="replace").strip()
            cameras.append((i, name or f"MVS Camera {i}"))
        except Exception:
            cameras.append((i, f"MVS Camera {i}"))

    return cameras


class MVSCamera:
    """Hikvision MVS camera with an OpenCV-compatible read()/release() interface.

    Returns BGR numpy arrays from read(), just like cv2.VideoCapture.
    """

    def __init__(self) -> None:
        self._lib = _load_mvs()
        self._handle = c_void_p(None)
        self._opened = False
        self._grabbing = False
        self._sdk_lock = threading.RLock()
        self.last_error: str = ""   # human-readable failure reason

    def open(self, index: int = 0) -> bool:
        """Open MVS camera by index (from enumerate_mvs_cameras). Returns True on success."""
        with self._sdk_lock:
            if self._lib is None:
                self.last_error = "MvCameraControl.dll not found — MVS SDK not installed"
                return False

            device_list = MV_CC_DEVICE_INFO_LIST()
            ret = self._lib.MV_CC_EnumDevices(
                MV_GIGE_DEVICE | MV_USB_DEVICE,
                ctypes.byref(device_list)
            )
            if ret != MV_OK:
                self.last_error = f"EnumDevices failed (code 0x{ret & 0xFFFFFFFF:08X})"
                return False
            if device_list.nDeviceNum == 0:
                self.last_error = "No MVS cameras found during open — was it unplugged?"
                return False
            if index >= device_list.nDeviceNum:
                self.last_error = f"Camera index {index} out of range (found {device_list.nDeviceNum})"
                return False

            ret = self._lib.MV_CC_CreateHandle(
                ctypes.byref(self._handle),
                device_list.pDeviceInfo[index]
            )
            if ret != MV_OK:
                self.last_error = f"CreateHandle failed (code 0x{ret & 0xFFFFFFFF:08X})"
                return False

            # Try access modes: Exclusive → Control (allows other monitors)
            open_ret = MV_OK + 1
            for mode in (MV_ACCESS_Exclusive, MV_ACCESS_Control):
                open_ret = self._lib.MV_CC_OpenDevice(
                    self._handle, ctypes.c_uint(mode), ctypes.c_ushort(0)
                )
                if open_ret == MV_OK:
                    break

            if open_ret != MV_OK:
                self._lib.MV_CC_DestroyHandle(self._handle)
                self._handle = c_void_p(None)
                self.last_error = (
                    f"OpenDevice failed (code 0x{open_ret & 0xFFFFFFFF:08X}).\n"
                    "Likely causes:\n"
                    "  1. MVS Viewer is open — close it and try again\n"
                    "  2. iVMS-4200 or any other Hikvision app is using the camera\n"
                    "  3. Unplug and replug the USB cable, then click ⚙ Scan MVS again"
                )
                return False

            self._opened = True
            self._grabbing = False
            self.last_error = ""
            return True

    def isOpened(self) -> bool:
        return self._opened

    def read(self, timeout_ms: int = 1000) -> tuple[bool, "np.ndarray | None"]:
        """Grab one frame. Returns (True, bgr_array) or (False, None).

        Args:
            timeout_ms: How long to wait for a frame (default 1000 ms).
                        Use a short value (e.g. 80) when the camera is in
                        hardware-trigger mode so the main UI thread is not
                        blocked for a full second on each empty poll.
        """
        with self._sdk_lock:
            if not self._opened or self._lib is None:
                return False, None

            if not self._grabbing:
                ret = self._lib.MV_CC_StartGrabbing(self._handle)
                if ret != MV_OK:
                    self.last_error = f"StartGrabbing failed (code 0x{ret & 0xFFFFFFFF:08X})"
                    return False, None
                self._grabbing = True

            frame_out = MV_FRAME_OUT()
            ctypes.memset(ctypes.byref(frame_out), 0, ctypes.sizeof(frame_out))
            ret = self._lib.MV_CC_GetImageBuffer(
                self._handle, ctypes.byref(frame_out), max(1, int(timeout_ms))
            )
            if ret != MV_OK:
                return False, None

            try:
                w      = frame_out.stFrameInfo.nWidth
                h      = frame_out.stFrameInfo.nHeight
                n      = frame_out.stFrameInfo.nFrameLen
                ptype  = frame_out.stFrameInfo.enPixelType

                if n == 0 or frame_out.pBufAddr is None:
                    return False, None

                raw_bytes = ctypes.string_at(frame_out.pBufAddr, n)
                data = np.frombuffer(raw_bytes, dtype=np.uint8)

                frame = self._convert(data, w, h, ptype)
                return (frame is not None), frame
            except Exception:
                return False, None
            finally:
                self._lib.MV_CC_FreeImageBuffer(
                    self._handle, ctypes.byref(frame_out)
                )

    def _set_enum_value(self, key: str, value: int) -> bool:
        if not self._opened or self._lib is None:
            self.last_error = "MVS camera is not open"
            return False
        try:
            ret = self._lib.MV_CC_SetEnumValue(
                self._handle,
                ctypes.c_char_p(key.encode("ascii")),
                ctypes.c_uint(int(value)),
            )
        except Exception as exc:
            self.last_error = f"SetEnumValue({key}) failed: {exc}"
            return False
        if ret != MV_OK:
            self.last_error = (
                f"SetEnumValue({key}={value}) failed "
                f"(0x{ret & 0xFFFFFFFF:08X})"
            )
            return False
        self.last_error = ""
        return True

    def _set_enum_value_by_string(self, key: str, value: str) -> bool:
        if not self._opened or self._lib is None:
            self.last_error = "MVS camera is not open"
            return False
        fn = getattr(self._lib, "MV_CC_SetEnumValueByString", None)
        if fn is None:
            self.last_error = "SetEnumValueByString not available in this SDK build"
            return False
        try:
            ret = fn(
                self._handle,
                ctypes.c_char_p(key.encode("ascii")),
                ctypes.c_char_p(value.encode("ascii")),
            )
        except Exception as exc:
            self.last_error = f"SetEnumValueByString({key}) failed: {exc}"
            return False
        if ret != MV_OK:
            self.last_error = (
                f"SetEnumValueByString({key}={value}) failed "
                f"(0x{ret & 0xFFFFFFFF:08X})"
            )
            return False
        self.last_error = ""
        return True

    def _set_bool_value(self, key: str, value: bool) -> bool:
        if not self._opened or self._lib is None:
            self.last_error = "MVS camera is not open"
            return False
        try:
            ret = self._lib.MV_CC_SetBoolValue(
                self._handle,
                ctypes.c_char_p(key.encode("ascii")),
                ctypes.c_bool(bool(value)),
            )
        except Exception as exc:
            self.last_error = f"SetBoolValue({key}) failed: {exc}"
            return False
        if ret != MV_OK:
            self.last_error = (
                f"SetBoolValue({key}={int(bool(value))}) failed "
                f"(0x{ret & 0xFFFFFFFF:08X})"
            )
            return False
        self.last_error = ""
        return True

    def _get_node_access_mode(self, key: str) -> int | None:
        """Return MV_XML_AccessMode for a node, or None if unavailable."""
        if not self._opened or self._lib is None:
            self.last_error = "MVS camera is not open"
            return None
        fn = getattr(self._lib, "MV_XML_GetNodeAccessMode", None)
        if fn is None:
            self.last_error = "MV_XML_GetNodeAccessMode not available in this SDK build"
            return None
        mode = ctypes.c_int(-1)
        try:
            ret = fn(
                self._handle,
                ctypes.c_char_p(key.encode("ascii")),
                ctypes.byref(mode),
            )
        except Exception as exc:
            self.last_error = f"GetNodeAccessMode({key}) failed: {exc}"
            return None
        if ret != MV_OK:
            self.last_error = (
                f"GetNodeAccessMode({key}) failed "
                f"(0x{ret & 0xFFFFFFFF:08X})"
            )
            return None
        self.last_error = ""
        return int(mode.value)

    @staticmethod
    def _access_mode_name(mode: int | None) -> str:
        names = {
            AM_NI: "NI",
            AM_NA: "NA",
            AM_WO: "WO",
            AM_RO: "RO",
            AM_RW: "RW",
        }
        if mode is None:
            return "ERR"
        return names.get(int(mode), str(int(mode)))

    def io_node_access_summary(self) -> dict[str, str]:
        """Return access-mode summary for line I/O related nodes."""
        with self._sdk_lock:
            keys = (
                "TriggerMode",
                "TriggerSource",
                "LineSelector",
                "LineMode",
                "LineSource",
                "UserOutputSelector",
                "UserOutputValue",
            )
            summary: dict[str, str] = {}
            for key in keys:
                mode = self._get_node_access_mode(key)
                summary[key] = self._access_mode_name(mode)
            return summary

    def supported_user_output_selectors(self, max_index: int = 8) -> list[int]:
        """Probe which UserOutputSelector indices are supported (trial-based)."""
        with self._sdk_lock:
            if not self._opened or self._lib is None:
                self.last_error = "MVS camera is not open"
                return []

            supported: list[int] = []
            for idx in range(max(0, int(max_index))):
                if self._set_enum_value_by_string("UserOutputSelector", f"UserOutput{idx}") \
                        or self._set_enum_value("UserOutputSelector", idx):
                    supported.append(idx)

            if supported:
                # Leave selector at a valid value to avoid stale invalid selection.
                if not self._set_enum_value_by_string(
                    "UserOutputSelector", f"UserOutput{supported[0]}"
                ):
                    self._set_enum_value("UserOutputSelector", supported[0])
                self.last_error = ""
            elif self._set_bool_value("UserOutputValue", False):
                # Some cameras expose a single output value but selector is not writable.
                supported = [0]
                self.last_error = ""
            else:
                self.last_error = (
                    f"No supported UserOutputSelector found in 0..{max(0, int(max_index)) - 1}"
                )
            return supported

    def enable_line_trigger(self, trigger_source: int = TRIGGER_SOURCE_LINE0) -> bool:
        """Enable hardware trigger mode from a camera input line."""
        with self._sdk_lock:
            if not self._set_enum_value("TriggerSource", int(trigger_source)):
                return False
            return self._set_enum_value("TriggerMode", TRIGGER_MODE_ON)

    def disable_trigger(self) -> bool:
        """Disable hardware trigger mode (free-run acquisition)."""
        with self._sdk_lock:
            return self._set_enum_value("TriggerMode", TRIGGER_MODE_OFF)

    def set_user_output(self, output_index: int, state: bool) -> bool:
        """Set camera UserOutput value (requires camera-side line mapping)."""
        with self._sdk_lock:
            idx = int(output_index)
            selector_ok = self._set_enum_value_by_string("UserOutputSelector", f"UserOutput{idx}")
            selector_err = self.last_error if not selector_ok else ""
            if not selector_ok:
                selector_ok = self._set_enum_value("UserOutputSelector", idx)
                if not selector_ok:
                    selector_err = self.last_error or selector_err

            # Try writing the output value even if selector write fails.
            # On some models selector is fixed/locked while UserOutputValue is writable.
            if self._set_bool_value("UserOutputValue", bool(state)):
                return True

            value_err = self.last_error
            if selector_err and value_err:
                self.last_error = f"{selector_err}; {value_err}"
            elif selector_err:
                self.last_error = selector_err
            else:
                self.last_error = value_err
            return False

    def pulse_user_output(self, output_index: int, pulse_ms: int = 80) -> bool:
        """Pulse selected UserOutput HIGH for pulse_ms then LOW."""
        with self._sdk_lock:
            if not self.set_user_output(output_index, True):
                return False
        ok = True
        try:
            time.sleep(max(0.0, float(pulse_ms)) / 1000.0)
        finally:
            with self._sdk_lock:
                ok = self.set_user_output(output_index, False) and ok
        return ok

    def _select_line_selector(self, line: str) -> bool:
        ok = self._set_enum_value_by_string("LineSelector", line)
        if ok:
            return True
        idx = {"Line0": 0, "Line1": 1, "Line2": 2, "Line3": 3}.get(str(line), 1)
        return self._set_enum_value("LineSelector", idx)

    def strobe_line_set_state(
        self,
        line: str = "Line1",
        state: bool = False,
        line_mode: str = "Strobe",
        line_source: str = "FrameTriggerWait",
    ) -> bool:
        """Set selected strobe line HIGH/LOW directly."""
        with self._sdk_lock:
            if not self._opened or self._lib is None:
                self.last_error = "MVS camera is not open"
                return False

            if not self._select_line_selector(line):
                return False
            self._set_enum_value_by_string("LineMode", line_mode)
            self._set_enum_value_by_string("LineSource", line_source)
            return self._set_bool_value("StrobeEnable", bool(state))

    def strobe_line_pulse(
        self,
        line: str = "Line1",
        pulse_ms: int = 2000,
        line_mode: str = "Strobe",
        line_source: str = "FrameTriggerWait",
    ) -> bool:
        """Pulse an output line HIGH using the camera StrobeEnable feature.

        Mirrors the working MATLAB approach exactly:
            LineSelector = line          (e.g. "Line1" or "Line2")
            LineMode     = line_mode     ("Strobe" for output lines)
            LineSource   = line_source   ("FrameTriggerWait")
            StrobeEnable = True          → pin goes HIGH
            sleep(pulse_ms / 1000)
            StrobeEnable = False         → pin goes LOW

        No TriggerMode changes are needed.
        """
        if not self.strobe_line_set_state(
            line=line,
            state=True,
            line_mode=line_mode,
            line_source=line_source,
        ):
            return False

        # Hold the pulse outside the lock so the preview loop is not stalled.
        time.sleep(max(0.01, float(pulse_ms)) / 1000.0)

        # Drive the pin LOW.
        with self._sdk_lock:
            if not self._opened or self._lib is None:
                return True  # camera closed during sleep — treat as success

        ok = self.strobe_line_set_state(
            line=line,
            state=False,
            line_mode=line_mode,
            line_source=line_source,
        )
        if ok:
            self.last_error = ""
        return ok

    def send_software_trigger(
        self, restore_source: int = TRIGGER_SOURCE_LINE0
    ) -> bool:
        """Pulse Line1 via TriggerSoftware (SoftTriggerActive workaround).

        Prerequisites (set once in MVS Viewer):
            LineSelector = Line1
            LineSource   = SoftTriggerActive

        This method:
          1. Stops any live grab that is running.
          2. Sets TriggerMode=ON + TriggerSource=Software (required for the
             command to be accepted by the camera firmware).
          3. Fires TriggerSoftware  →  Line1 pulses HIGH, camera grabs 1 frame.
          4. Reads and discards the dummy frame so it never reaches the app.
          5. Restores TriggerMode=ON + TriggerSource=restore_source in the
             finally-block.  Keeping TriggerMode ON (not free-run/OFF) is
             critical: in free-run mode the camera fires SoftTriggerActive for
             every frame it grabs, which would pulse Line1 continuously and
             flood the PLC with spurious signals.

        Args:
            restore_source: TRIGGER_SOURCE_* to switch back to after the pulse
                            (default LINE0 = hardware trigger from PLC).
        """
        with self._sdk_lock:
            if not self._opened or self._lib is None:
                self.last_error = "MVS camera is not open"
                return False

            fn_cmd = getattr(self._lib, "MV_CC_SetCommandValue", None)
            if fn_cmd is None:
                self.last_error = "MV_CC_SetCommandValue not available in this SDK build"
                return False

            # 1. Stop live grabbing (leaves camera in idle state).
            if self._grabbing:
                self._lib.MV_CC_StopGrabbing(self._handle)
                self._grabbing = False

            try:
                # 2. Enable triggered mode with software source.
                if not self._set_enum_value("TriggerMode", TRIGGER_MODE_ON):
                    return False
                if not self._set_enum_value("TriggerSource", TRIGGER_SOURCE_SOFTWARE):
                    return False

                # Start grabbing for one triggered frame.
                ret = self._lib.MV_CC_StartGrabbing(self._handle)
                if ret != MV_OK:
                    self.last_error = (
                        f"StartGrabbing failed before soft trigger "
                        f"(0x{ret & 0xFFFFFFFF:08X})"
                    )
                    return False
                self._grabbing = True

                # 3. Fire TriggerSoftware → pulses Line1.
                try:
                    ret = fn_cmd(
                        self._handle,
                        ctypes.c_char_p(b"TriggerSoftware"),
                    )
                except Exception as exc:
                    self.last_error = f"SetCommandValue(TriggerSoftware) raised: {exc}"
                    return False
                if ret != MV_OK:
                    self.last_error = (
                        f"SetCommandValue(TriggerSoftware) failed "
                        f"(0x{ret & 0xFFFFFFFF:08X})"
                    )
                    return False

                # 4. Drain the dummy frame so it never enters the live preview.
                #    100 ms is plenty — the frame is ready within a few ms of
                #    the TriggerSoftware command. Keeping this short reduces
                #    the time _sdk_lock is held and prevents the main UI thread
                #    from stalling when it next tries to call read().
                dummy = MV_FRAME_OUT()
                ctypes.memset(ctypes.byref(dummy), 0, ctypes.sizeof(dummy))
                drain_ret = self._lib.MV_CC_GetImageBuffer(
                    self._handle, ctypes.byref(dummy), 100
                )
                if drain_ret == MV_OK:
                    self._lib.MV_CC_FreeImageBuffer(self._handle, ctypes.byref(dummy))

                self.last_error = ""
                return True

            finally:
                # 5. Restore to hardware-trigger-wait mode (TriggerMode=ON).
                #    NEVER restore to TriggerMode=OFF (free-run): in free-run
                #    mode SoftTriggerActive fires Line1 on every grabbed frame.
                if self._grabbing:
                    self._lib.MV_CC_StopGrabbing(self._handle)
                    self._grabbing = False
                self._set_enum_value("TriggerSource", int(restore_source))
                self._set_enum_value("TriggerMode", TRIGGER_MODE_ON)
                # The preview loop read() will call StartGrabbing; it will
                # block on GetImageBuffer until the next hardware trigger
                # arrives from the PLC — no frames, no spurious pulses.

    def _convert(
        self,
        data: "np.ndarray",
        w: int,
        h: int,
        ptype: int,
    ) -> "np.ndarray | None":
        """Convert raw pixel data to a BGR uint8 numpy array."""
        if not _HAS_CV2:
            return None

        try:
            if ptype == PixelType_Gvsp_Mono8:
                gray = data[:h * w].reshape(h, w)
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            elif ptype == PixelType_Gvsp_BGR8_Packed:
                return data[:h * w * 3].reshape(h, w, 3).copy()

            elif ptype == PixelType_Gvsp_RGB8_Packed:
                rgb = data[:h * w * 3].reshape(h, w, 3)
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            elif ptype == PixelType_Gvsp_BayerGR8:
                return cv2.cvtColor(data[:h * w].reshape(h, w), cv2.COLOR_BayerGR2BGR)

            elif ptype == PixelType_Gvsp_BayerRG8:
                return cv2.cvtColor(data[:h * w].reshape(h, w), cv2.COLOR_BayerRG2BGR)

            elif ptype == PixelType_Gvsp_BayerGB8:
                return cv2.cvtColor(data[:h * w].reshape(h, w), cv2.COLOR_BayerGB2BGR)

            elif ptype == PixelType_Gvsp_BayerBG8:
                return cv2.cvtColor(data[:h * w].reshape(h, w), cv2.COLOR_BayerBG2BGR)

            elif ptype in (PixelType_Gvsp_YUV422_Packed, PixelType_Gvsp_YUV422_YUYV):
                yuv = data[:h * w * 2].reshape(h, w, 2)
                return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_YUYV)

            else:
                # Unknown format – attempt mono fallback
                if len(data) >= h * w:
                    gray = data[:h * w].reshape(h, w)
                    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                return None

        except Exception:
            return None

    def release(self) -> None:
        """Stop grabbing and close the camera."""
        with self._sdk_lock:
            if self._lib is None:
                return
            if self._grabbing:
                self._lib.MV_CC_StopGrabbing(self._handle)
                self._grabbing = False
            if self._opened:
                self._lib.MV_CC_CloseDevice(self._handle)
                self._lib.MV_CC_DestroyHandle(self._handle)
                self._opened = False
                self._handle = c_void_p(None)
