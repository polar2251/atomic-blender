"""UI-agnostic Undo / Redo snapshot history stack."""

from __future__ import annotations

import copy
from typing import Any


class UndoHistory:
    """Bounded snapshot stack supporting duplicate suppression and branch truncation."""

    def __init__(self, max_depth: int = 50) -> None:
        self._max_depth = max(2, int(max_depth))
        self._states: list[Any] = []
        self._index: int = -1

    def __len__(self) -> int:
        return len(self._states)

    @property
    def can_undo(self) -> bool:
        return self._index > 0

    @property
    def can_redo(self) -> bool:
        return 0 <= self._index < len(self._states) - 1

    def clear(self) -> None:
        """Clear all recorded history states."""
        self._states.clear()
        self._index = -1

    def current(self) -> Any | None:
        """Return a deep copy of the active snapshot, or ``None`` if empty."""
        if 0 <= self._index < len(self._states):
            return copy.deepcopy(self._states[self._index])
        return None

    def push(self, snapshot: Any) -> bool:
        """Record a snapshot; return False if identical to the current active snapshot."""
        snap = copy.deepcopy(snapshot)
        if 0 <= self._index < len(self._states) and snap == self._states[self._index]:
            return False
        if self._index + 1 < len(self._states):
            del self._states[self._index + 1 :]
        self._states.append(snap)
        overflow = len(self._states) - self._max_depth
        if overflow > 0:
            del self._states[:overflow]
        self._index = len(self._states) - 1
        return True

    def undo(self) -> Any | None:
        """Step back one snapshot and return a deep copy, or ``None`` if at earliest state."""
        if not self.can_undo:
            return None
        self._index -= 1
        return copy.deepcopy(self._states[self._index])

    def redo(self) -> Any | None:
        """Step forward one snapshot and return a deep copy, or ``None`` if at latest state."""
        if not self.can_redo:
            return None
        self._index += 1
        return copy.deepcopy(self._states[self._index])
