"""
Rolling Progress Reporter
=========================

A tiny, dependency-free progress reporter for the long-running parts of
the evaluation.

Why this module exists
----------------------
The attribute agent samples 20 questions × 10 rounds = 200 graph builds
by default.  Each build loads a spaCy model, runs an embedder over the
sentences, and encodes every attribute descriptor.  Without any output
the terminal shows a sentence-transformers "Batches: ..." progress bar
that restarts for every question — which reads as a hung run, not as
work in progress.  The user cannot tell how many questions are left, how
many gaps have been found, or which attributes have already been rolled
into the table.

This reporter replaces that with a single line that states exactly
those three things:

    [agent] 137/200 draws | gaps 5 | accepted 2 | 3 attrs done | ETA 0:42

Design constraints
------------------
* **No new dependency.**  Pure stdlib.
* **One line, not a scrolling log.**  The line is rewritten in place
  with ANSI cursor movement when stdout is a TTY, and printed as
  ordinary lines when it is not (piped to a file, or a CI log).
* **Never breaks the run.**  Every write is wrapped; a broken pipe or a
  weird console must not kill the agent.
* **Windows-safe.**  The Windows console needs ``\\r`` (not ANSI) for
  in-place rewriting, and it cannot handle the full ANSI cursor
  movement.  We use ``\\r`` + spaces-to-erase, which works on every
  platform.
"""

from __future__ import annotations

import shutil
import sys
import time
from typing import Any, Dict, List, Optional, TextIO


def _fmt_eta(seconds: float) -> str:
    """Format a duration as `M:SS` (or `H:MM:SS` past an hour)."""
    if seconds < 0 or seconds != seconds:  # NaN / negative
        return "--:--"
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"
    return f"{seconds // 60}:{seconds % 60:02d}"


class ProgressReporter:
    """
    Base reporter — does nothing.

    Subclasses implement `update` / `finish`.  The agent only ever
    calls those two methods, so any object with that shape works.
    """

    def update(self, done: int, total: int, note: str = "") -> None:
        """Called after each unit of work."""

    def finish(self, summary: Optional[Dict[str, Any]] = None) -> None:
        """Called once when the work is complete."""

    def section(self, title: str) -> None:
        """Announce a new phase."""


class NullReporter(ProgressReporter):
    """Does nothing.  Used when progress output is not wanted."""


class ConsoleReporter(ProgressReporter):
    """
    Prints a single updating line to the console.

    Args:
        prefix:    Label shown before the counters (e.g. "agent").
        stream:    Where to write.  Defaults to `sys.stdout`.
        enabled:   When False the reporter is silent.
        min_interval:
            Minimum seconds between two redraws.  The agent calls
            `update()` once per question, which is often several times
            per second; redrawing that fast is unreadable AND slows the
            run down on a Windows console.
    """

    def __init__(
        self,
        prefix: str = "agent",
        stream: Optional[TextIO] = None,
        enabled: bool = True,
        min_interval: float = 0.25,
    ) -> None:
        self.prefix = prefix
        self.stream = stream if stream is not None else sys.stdout
        self.enabled = enabled
        self.min_interval = min_interval
        self._last_draw = 0.0
        self._last_width = 0
        self._start = time.time()
        self._extra: Dict[str, Any] = {}
        self._in_line = False

    # -- helpers --------------------------------------------------------

    def _write(self, text: str) -> None:
        """Write text to the stream, swallowing every error."""
        try:
            self.stream.write(text)
            self.stream.flush()
        except UnicodeEncodeError:
            # A GBK console cannot encode every character.  Degrade to
            # replacement characters rather than crashing the run.
            enc = getattr(self.stream, "encoding", None) or "utf-8"
            try:
                self.stream.write(
                    text.encode(enc, errors="replace").decode(enc, errors="replace")
                )
                self.stream.flush()
            except Exception:  # noqa: BLE001
                self.enabled = False
        except Exception:  # noqa: BLE001 - a console error must not kill the run
            self.enabled = False

    def _erase(self) -> None:
        """Erase the current line if we are mid-line."""
        if self._in_line:
            self._write("\r" + " " * self._last_width + "\r")
            self._in_line = False

    def _is_tty(self) -> bool:
        try:
            return bool(self.stream.isatty())
        except Exception:  # noqa: BLE001
            return False

    # -- API ------------------------------------------------------------

    def set_extra(self, **kwargs: Any) -> None:
        """
        Attach counters that are shown next to the progress line.

        The agent uses this for the rolling tallies: how many gaps have
        been found, how many attributes have been accepted, how many
        are already in the table.
        """
        self._extra.update(kwargs)

    def section(self, title: str) -> None:
        """Start a new phase on its own line."""
        self._erase()
        try:
            width = shutil.get_terminal_size((80, 24)).columns
        except Exception:  # noqa: BLE001
            width = 80
        self._write(f"[{self.prefix}] {title}\n")
        self._last_width = 0

    def update(self, done: int, total: int, note: str = "") -> None:
        """Redraw the progress line."""
        if not self.enabled:
            return
        now = time.time()
        if now - self._last_draw < self.min_interval:
            return
        self._last_draw = now

        pct = (done / total * 100.0) if total else 0.0
        elapsed = now - self._start
        remaining = (
            (elapsed / done) * (total - done) if done else 0.0
        )

        parts = [f"{done}/{total} ({pct:.0f}%)"]
        for key, value in self._extra.items():
            parts.append(f"{key} {value}")
        parts.append(f"ETA {_fmt_eta(remaining)}")
        line = f"[{self.prefix}] " + " | ".join(parts)
        if note:
            line += f" | {note}"

        # Keep the line short enough to be readable.
        try:
            width = shutil.get_terminal_size((80, 24)).columns
        except Exception:  # noqa: BLE001
            width = 80
        if len(line) > max(40, width - 1):
            line = line[:max(40, width - 4)] + "..."

        # Erase the previous line, then write the new one.
        self._write("\r" + " " * self._last_width + "\r" + line)
        self._last_width = len(line)
        self._in_line = True

    def finish(self, summary: Optional[Dict[str, Any]] = None) -> None:
        """Erase the progress line and print the final summary."""
        if not self.enabled:
            return
        self._erase()
        elapsed = time.time() - self._start
        parts = [f"{self.prefix} done in {_fmt_eta(elapsed)}"]
        for key, value in (summary or {}).items():
            parts.append(f"{key} {value}")
        self._write("[" + "] [".join(parts) + "]\n")
        self._last_width = 0


class MultiReporter(ProgressReporter):
    """Fans `update` / `finish` / `section` out to several reporters."""

    def __init__(self, reporters: List[ProgressReporter]) -> None:
        self.reporters = [r for r in reporters if r is not None]

    def update(self, done: int, total: int, note: str = "") -> None:
        for r in self.reporters:
            r.update(done, total, note)

    def finish(self, summary: Optional[Dict[str, Any]] = None) -> None:
        for r in self.reporters:
            r.finish(summary)

    def section(self, title: str) -> None:
        for r in self.reporters:
            r.section(title)

    def set_extra(self, **kwargs: Any) -> None:
        for r in self.reporters:
            setter = getattr(r, "set_extra", None)
            if callable(setter):
                setter(**kwargs)
