"""OS clipboard and file explorer helpers (UI-framework independent)."""

from __future__ import annotations

import io
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


def copy_pil_image_to_clipboard(img: Any) -> bool:
    """Copy a PIL Image to the Windows clipboard as CF_DIB. Returns True on success."""
    if img is None:
        return False
    if not (sys.platform.startswith("win") or os.name == "nt"):
        return False
    try:
        import ctypes
        from ctypes import wintypes

        buf = io.BytesIO()
        img.convert("RGB").save(buf, "BMP")
        dib = buf.getvalue()[14:]  # Strip 14-byte BITMAPFILEHEADER for CF_DIB
        if not dib:
            return False

        kernel32 = ctypes.windll.kernel32
        user32 = ctypes.windll.user32

        kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
        kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
        kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalLock.restype = wintypes.LPVOID
        kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalUnlock.restype = wintypes.BOOL
        kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalFree.restype = wintypes.HGLOBAL

        user32.OpenClipboard.argtypes = [wintypes.HWND]
        user32.OpenClipboard.restype = wintypes.BOOL
        user32.EmptyClipboard.argtypes = []
        user32.EmptyClipboard.restype = wintypes.BOOL
        user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
        user32.SetClipboardData.restype = wintypes.HANDLE
        user32.CloseClipboard.argtypes = []
        user32.CloseClipboard.restype = wintypes.BOOL

        gmem_moveable = 0x0002
        cf_dib = 8
        h_mem = kernel32.GlobalAlloc(gmem_moveable, len(dib))
        if not h_mem:
            return False
        ptr = kernel32.GlobalLock(h_mem)
        if not ptr:
            kernel32.GlobalFree(h_mem)
            return False
        ctypes.memmove(ptr, dib, len(dib))
        kernel32.GlobalUnlock(h_mem)

        if not user32.OpenClipboard(None):
            kernel32.GlobalFree(h_mem)
            return False
        try:
            user32.EmptyClipboard()
            if not user32.SetClipboardData(cf_dib, h_mem):
                kernel32.GlobalFree(h_mem)
                return False
            # Clipboard owns h_mem after SetClipboardData succeeds
            return True
        finally:
            user32.CloseClipboard()
    except Exception:
        return False


def open_folder(folder: Path) -> None:
    """Open ``folder`` in the OS file explorer."""
    if sys.platform.startswith("win") or os.name == "nt":
        os.startfile(str(folder))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(folder)])
    else:
        subprocess.Popen(["xdg-open", str(folder)])
