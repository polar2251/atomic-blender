"""Unit tests for UI-agnostic atomic_blender.core modules."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest

from atomic_blender import core, gui


class TestCoreModules(unittest.TestCase):
    def test_core_import_is_ui_and_ase_free(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, atomic_blender.core; "
                "assert not any(m == 'tkinter' or m.startswith('tkinter.') for m in sys.modules); "
                "assert not any(m == 'ase' or m.startswith('ase.') for m in sys.modules); "
                "print('CORE_CLEAN_OK')",
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("CORE_CLEAN_OK", proc.stdout)

    def test_undo_history_lifecycle_and_dedup(self) -> None:
        h = core.UndoHistory(max_depth=3)
        self.assertFalse(h.can_undo)
        self.assertFalse(h.can_redo)
        self.assertIsNone(h.current())

        self.assertTrue(h.push({"step": 1, "nested": [10]}))
        # Consecutive duplicate ignored
        self.assertFalse(h.push({"step": 1, "nested": [10]}))
        # Mutating caller dict does not mutate stored snapshot
        d2 = {"step": 2, "nested": [20]}
        self.assertTrue(h.push(d2))
        d2["nested"].append(999)
        self.assertEqual(h.current(), {"step": 2, "nested": [20]})

        self.assertTrue(h.push({"step": 3}))
        self.assertTrue(h.push({"step": 4}))
        # Bounded at max_depth=3 (states 2, 3, 4 kept)
        self.assertEqual(len(h), 3)

        self.assertEqual(h.undo(), {"step": 3})
        self.assertEqual(h.undo(), {"step": 2, "nested": [20]})
        self.assertFalse(h.can_undo)
        self.assertTrue(h.can_redo)

        # Pushing after undo truncates forward redo branch
        self.assertTrue(h.push({"step": 99}))
        self.assertFalse(h.can_redo)
        self.assertEqual(h.current(), {"step": 99})
        self.assertEqual(h.undo(), {"step": 2, "nested": [20]})
        self.assertEqual(h.redo(), {"step": 99})

        h.clear()
        self.assertEqual(len(h), 0)
        self.assertIsNone(h.current())

    def test_gui_reexports_core_symbols_identically(self) -> None:
        for sym in (
            "BlenderPreviewWorker",
            "GuiRenderConfig",
            "MATERIAL_FINISH_PRESETS",
            "UndoHistory",
            "auto_detect_adsorbate_indices",
            "build_render_cli_args",
            "compute_adsorbate_spotlight_alphas",
            "compute_atom_alphas",
            "compute_atom_radius_scales",
            "compute_depth_fade_alphas",
            "compute_preview_resolution",
            "compute_selection_measurement",
            "copy_pil_image_to_clipboard",
            "extract_input_path",
            "find_neighbor_atoms",
            "format_input_display",
            "inspect_structure_summary",
            "resolve_batch_output_paths",
            "select_path_strip_indices",
            "validate_gui_config",
        ):
            self.assertIs(getattr(gui, sym), getattr(core, sym), msg=f"Re-export mismatch for {sym}")


if __name__ == "__main__":
    unittest.main()
