"""Tests for PySide6 + OpenGL 3D desktop GUI."""

from __future__ import annotations

from pathlib import Path
import time
import unittest

from atomic_blender.cli import build_parser
from atomic_blender.core import JOURNAL_PRESETS
from atomic_blender.qt_gui import HAS_PYSIDE6


def _process_until(app, predicate, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    app.processEvents()
    return bool(predicate())


@unittest.skipUnless(HAS_PYSIDE6, "PySide6 is not installed")
class TestQtGui(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PySide6.QtWidgets import QApplication
        from atomic_blender.qt_gui.theme import apply_studio_theme

        cls.app = QApplication.instance() or QApplication([])
        apply_studio_theme(cls.app)

    def test_qt_window_welcome_orbit_preview_undo_journal_and_clipping(self) -> None:
        from PySide6.QtCore import QPointF, Qt
        from atomic_blender.qt_gui import AtomicStudioQtWindow

        repo_root = Path(__file__).resolve().parents[1]
        cu_cif = repo_root / "examples" / "fcc_cu_bulk.cif"
        self.assertTrue(cu_cif.is_file())

        win = AtomicStudioQtWindow(prompt_missing_blender=False)
        preview_calls: list[dict] = []
        win._preview_worker.request = lambda payload: preview_calls.append(dict(payload))  # type: ignore[method-assign]
        win.entry_blender.setText(str(cu_cif))  # existing file satisfies Blender check in mock preview test

        try:
            win.show()
            self.app.processEvents()

            # 1. Welcome screen & 3 sample pills
            self.assertIs(win.canvas_stack.currentWidget(), win.welcome_widget)
            pill_texts = [b.text() for b in win.welcome_widget.sample_buttons.values()]
            self.assertEqual(pill_texts, ["Cu Bulk", "Water Box", "CO / Pt(111)"])

            # 2. Load example CIF structure
            win.open_path(str(cu_cif))
            ok = _process_until(self.app, lambda: win._structure_cache is not None and win._vp_scene is not None)
            self.assertTrue(ok, "Structure inspect did not finish in time")
            self.assertIs(win.canvas_stack.currentWidget(), win.viewport)
            self.assertEqual(len(win._structure_cache["symbols"]), 108)
            self.assertGreater(len(preview_calls), 0)

            # 3. Switch View / Select / Scene tabs
            for tab_name, expected_idx in (("select", 1), ("scene", 2), ("view", 0)):
                win.switch_tab(tab_name)
                self.app.processEvents()
                self.assertEqual(win.tabs.currentIndex(), expected_idx)

            # 4. Simulate drag orbit + camera_settled -> triggers mocked BlenderPreviewWorker
            before_count = len(preview_calls)
            q_before = win._vp_scene.quat
            win.viewport.set_blender_preview(win.viewport.render_snapshot_pil() if hasattr(win.viewport, "render_snapshot_pil") else None)
            win.viewport.scene.orbit(0.25, -0.15)
            self.assertNotEqual(win._vp_scene.quat, q_before)
            win.viewport.clear_blender_preview()
            self.assertFalse(win.viewport.has_blender_preview())
            win.viewport.camera_settled.emit()
            self.app.processEvents()
            self.assertGreater(len(preview_calls), before_count)

            # 5. Undo / Redo stack restores camera quaternion
            win.undo()
            self.app.processEvents()
            self.assertEqual(tuple(round(v, 5) for v in win._vp_scene.quat), tuple(round(v, 5) for v in q_before))
            win.redo()
            self.app.processEvents()
            self.assertNotEqual(tuple(round(v, 5) for v in win._vp_scene.quat), tuple(round(v, 5) for v in q_before))

            # 6. Journal DPI presets update resolution & target DPI
            win.switch_tab("scene")
            self.assertEqual(win.combo_resolution.count(), len(JOURNAL_PRESETS))
            win.combo_resolution.setCurrentText("ACS 1-Col (600 DPI)")
            self.app.processEvents()
            self.assertEqual((win._res_w, win._res_h, win._target_dpi), JOURNAL_PRESETS["ACS 1-Col (600 DPI)"])

            # 7. Check CLI parser --backend flag
            parsed = build_parser().parse_args(["gui", "--backend", "qt"])
            self.assertEqual(parsed.backend, "qt")

            # 8. Zero widget clipping inside sidebar & toolbar at 1024x768 and 1440x900
            for width, height in ((1024, 768), (1440, 900)):
                win.resize(width, height)
                self.app.processEvents()
                for btn in (win.btn_render, win.btn_open, win.btn_fit, win.btn_save_view, win.btn_copy, win.btn_undo, win.btn_redo):
                    self.assertTrue(btn.isVisible())
                    self.assertGreaterEqual(btn.width(), btn.sizeHint().width() - 4, f"Clipped width on {btn.text()} at {width}x{height}")
                    self.assertGreaterEqual(btn.height(), btn.sizeHint().height() - 2, f"Clipped height on {btn.text()} at {width}x{height}")
        finally:
            win.close()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
