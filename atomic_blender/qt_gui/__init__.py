"""PySide6 + OpenGL GPU viewport desktop UI for Atomic Blender Studio."""

from __future__ import annotations

import sys

try:
    import PySide6  # noqa: F401
    from PySide6.QtOpenGLWidgets import QOpenGLWidget  # noqa: F401

    HAS_PYSIDE6 = True
except Exception:
    HAS_PYSIDE6 = False

if HAS_PYSIDE6:
    from atomic_blender.qt_gui.gl_viewport import AtomicGLViewport
    from atomic_blender.qt_gui.window import AtomicStudioQtWindow
else:
    AtomicGLViewport = None  # type: ignore[assignment,misc]
    AtomicStudioQtWindow = None  # type: ignore[assignment,misc]


def launch_qt_gui(initial_input: str | None = None) -> int:
    """Launch the PySide6 + OpenGL Atomic Blender Studio desktop application."""
    if not HAS_PYSIDE6:
        raise RuntimeError("PySide6 is not installed. Install with: pip install 'atomic-blender[qt]'")

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QIcon, QSurfaceFormat
    from PySide6.QtWidgets import QApplication

    from atomic_blender.cli import ensure_app_icon
    from atomic_blender.qt_gui.theme import apply_studio_theme

    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
        app = QApplication(sys.argv[:1])

    app.setApplicationName("Atomic Blender Studio")
    app.setOrganizationName("AtomicBlenderStudio")
    apply_studio_theme(app)

    try:
        ico_path = ensure_app_icon()
        if ico_path.is_file():
            app.setWindowIcon(QIcon(str(ico_path)))
    except Exception:
        pass

    win = AtomicStudioQtWindow(initial_input=initial_input, prompt_missing_blender=True)
    win.show()
    if owns_app:
        return int(app.exec())
    return 0


__all__ = [
    "AtomicGLViewport",
    "AtomicStudioQtWindow",
    "HAS_PYSIDE6",
    "launch_qt_gui",
]
