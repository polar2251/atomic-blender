"""Background Blender Cycles preview server manager."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import threading
from typing import Any

from atomic_blender.cli import package_dir


class BlenderPreviewWorker:
    """Coalescing persistent Blender subprocess for fast low-res camera previews."""

    def __init__(self, on_done: Any, on_error: Any, on_status: Any = None) -> None:
        self._on_done = on_done
        self._on_error = on_error
        self._on_status = on_status
        self._cond = threading.Condition()
        self._pending: dict[str, Any] | None = None
        self._stopped = False
        self._proc: subprocess.Popen[str] | None = None
        self._active_blender: str | None = None
        self._thread: threading.Thread | None = None

    def request(self, payload: dict[str, Any]) -> None:
        """Queue the latest preview request, superseding any unstarted prior request."""
        with self._cond:
            if self._stopped:
                return
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._loop, daemon=True)
                self._thread.start()
            self._pending = dict(payload)
            self._cond.notify_all()

    @staticmethod
    def _cleanup_popen(proc: subprocess.Popen[str]) -> None:
        """Terminate ``proc`` if still alive, reap it, and close all stdio pipes cleanly."""
        if getattr(proc, "stdin", None) is not None:
            try:
                proc.stdin.close()  # type: ignore[union-attr]
            except Exception:
                pass
        try:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=1.5)
        except Exception:
            pass
        for pipe in (getattr(proc, "stdout", None), getattr(proc, "stderr", None)):
            if pipe is not None:
                try:
                    pipe.close()
                except Exception:
                    pass

    def _close_proc(self) -> None:
        proc = self._proc
        self._proc = None
        self._active_blender = None
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                proc.stdin.write(json.dumps({"action": "quit"}) + "\n")
                proc.stdin.flush()
                proc.stdin.close()
                proc.wait(timeout=1.5)
        except Exception:
            pass
        finally:
            self._cleanup_popen(proc)

    def _ensure_proc(self, blender_exe: str) -> subprocess.Popen[str]:
        if self._proc is not None and self._proc.poll() is None and self._active_blender == blender_exe:
            return self._proc
        self._close_proc()
        pkg_dir = package_dir()
        scene_script = pkg_dir / "scene_builder.py"
        env = os.environ.copy()
        pkg_parent = str(pkg_dir.parent)
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{pkg_parent}{os.pathsep}{existing_pp}" if existing_pp else pkg_parent
        kwargs: dict[str, Any] = {}
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if creationflags:
            kwargs["creationflags"] = creationflags
        proc = subprocess.Popen(
            [
                blender_exe,
                "--background",
                "--factory-startup",
                "--python",
                str(scene_script),
                "--",
                "--preview-server",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            **kwargs,
        )
        assert proc.stdout is not None
        ready = False
        for line in proc.stdout:
            if line.strip() == "PREVIEW_READY":
                ready = True
                break
        if not ready:
            self._cleanup_popen(proc)
            raise RuntimeError("Blender preview server exited before becoming ready.")
        self._proc = proc
        self._active_blender = blender_exe
        return proc

    def _loop(self) -> None:
        while True:
            with self._cond:
                while not self._stopped and self._pending is None:
                    self._cond.wait()
                if self._stopped:
                    break
                cmd = self._pending
                self._pending = None
            assert cmd is not None
            seq = int(cmd["seq"])
            try:
                if self._on_status is not None:
                    self._on_status(seq, "Rendering Blender preview...")
                proc = self._ensure_proc(str(cmd["blender_exe"]))
                assert proc.stdin is not None and proc.stdout is not None
                wire_cmd = {k: v for k, v in cmd.items() if k != "blender_exe"}
                proc.stdin.write(json.dumps(wire_cmd) + "\n")
                proc.stdin.flush()

                finished = False
                for raw_line in proc.stdout:
                    line = raw_line.strip()
                    if line.startswith("PREVIEW_DONE "):
                        parts = line.split(" ", 2)
                        done_seq = int(parts[1])
                        out_path = Path(parts[2])
                        self._on_done(done_seq, out_path)
                        finished = True
                        break
                    if line.startswith("PREVIEW_ERR "):
                        parts = line.split(" ", 2)
                        err_seq = int(parts[1])
                        msg = parts[2] if len(parts) > 2 else "Unknown error"
                        self._on_error(err_seq, msg)
                        finished = True
                        break
                if not finished:
                    self._close_proc()
                    self._on_error(seq, "Blender preview server terminated unexpectedly.")
            except Exception as exc:
                self._close_proc()
                self._on_error(seq, str(exc))
        self._close_proc()

    def stop(self) -> None:
        """Terminate the background Blender preview subprocess cleanly."""
        with self._cond:
            self._stopped = True
            self._pending = None
            self._cond.notify_all()
        self._close_proc()
