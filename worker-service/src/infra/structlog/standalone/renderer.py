import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Iterator, MutableMapping

import structlog
from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.style import Style
from rich.text import Text

from src.infra.structlog.standalone.loaders import get_random_loader
from src.infra.structlog.standalone.timer import Timer

_UNSET = object()
END_SECTION = object()

_GREY = Style(color="rgb(150,150,150)")
_WHITE = Style(color="bright_white")
_CORAL = Style(color="rgb(255,127,80)")
_GREEN = Style(color="rgb(80,200,120)")
_RED = Style(color="rgb(220,60,60)")
_YELLOW = Style(color="rgb(255,200,50)")

_INDENT = "   "
_SEP = " · "

_RESERVED = {"event", "level", "timestamp", "task_name", "exc_info"}

_MINUTE = 60
_HOUR = 60 * 60
_DAY = 60 * 60 * 24


def format_elapsed(elapsed: float) -> str:
    if elapsed < _MINUTE:
        return f"{elapsed:.1f}s"
    elapsed = int(elapsed)
    if elapsed > _DAY:
        return f"{elapsed // _DAY}d {elapsed % _DAY // _HOUR}h"
    if elapsed > _HOUR:
        return f"{elapsed // _HOUR}h {elapsed % _HOUR // _MINUTE}m"
    return f"{elapsed // _MINUTE}m {elapsed % _MINUTE}s"


@dataclass
class _Message:
    text: str
    level: str
    attrs: dict[str, Any]
    elapsed: float = 0.0
    traceback: str | None = None


class _ActiveView:
    """Renderable that Rich's Live calls on every refresh.

    Re-reads renderer state under the renderer's lock and yields the
    current group-line + active-message-line.
    """

    def __init__(self, renderer: "TerminalRenderer") -> None:
        self._r = renderer

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        with self._r._lock:
            if not self._r._has_active or not self._r._messages:
                return
            last = self._r._messages[-1]
            spinner = next(self._r._loader)

            group = Text()
            group.append(f" {spinner} ", style=_CORAL)
            group.append(self._r._task_label(), style=_CORAL)
            yield group

            details = self._r.create_details(last.attrs)
            working = time.monotonic() - self._r._message_start
            msg = Text()
            msg.append(_INDENT)
            msg.append(last.text, style=_GREY)
            msg.append(
                f" ({self._r._format_inline(working, details)})", style=_GREY)
            yield msg


class TerminalRenderer:
    """Structlog processor that renders log events as a live CLI report.

    Layout (printed top-to-bottom, top is oldest):
      [ascii logo — printed once]
      [completed section close-lines, one each]
       (loader) task_name              ← live, refreshed by rich.Live
          message (working_time · details)

    The task_name is "sticky": once set, it persists across subsequent
    events that don't carry one. Only an explicit *different* task_name
    opens a new section. On task_name change:
      * success — close with one green ✔ line
      * failure (any error in section) — close with red ✗ line, then dump
                                          every collected message + tb.

    Uses rich.live.Live for the live area so that external writes
    (warnings, library prints) don't corrupt cursor positioning — Rich's
    Live redirects stdout/stderr and re-renders the live block beneath
    any out-of-band output.
    """

    def __init__(self) -> None:
        self._task_name: Any = _UNSET
        self._messages: list[_Message] = []
        self._failed: bool = False

        self._section_timer = Timer()
        self._message_start: float = 0.0

        self._lock = threading.RLock()

        self._console = Console(highlight=False, markup=False)
        self._loader: Iterator[str] = get_random_loader()

        self._has_active = False
        self._logo_done = False
        self._live_started = False
        self._live = Live(
            _ActiveView(self),
            console=self._console,
            refresh_per_second=10,
            transient=False,
            auto_refresh=True,
            redirect_stdout=True,
            redirect_stderr=True,
        )

    def create_logo(self) -> list[str]:
        """ASCII logo printed once at the very top."""
        return [
            "                    ",
            "   ╭──────────────╮ ",
            "   │              │ ",
            "   │   █▀█ █▀█    │ ",
            "   │   █▀█ █▀▀    │ ",
            "   │   ▀ ▀ ▀      │ ",
            "   │              │ ",
            "   │  ANIMATION   │ ",
            "   │  PIPELINE    │ ",
            "   │              │ ",
            "   ╰──────────────╯ ",
            "                    ",
        ]

    def create_details(self, attrs: dict[str, Any]) -> list[str]:
        """Turn event attrs into `key=value` chips for the inline area."""
        return [f"{k}={v}" for k, v in attrs.items() if k not in _RESERVED]

    def _format_inline(self, elapsed: float, details: list[str]) -> str:
        return _SEP.join([format_elapsed(elapsed), *details])

    def _task_label(self) -> str:
        if self._task_name is _UNSET or self._task_name is None:
            return ""
        return str(self._task_name)

    def _print_logo(self) -> None:
        for line in self.create_logo():
            self._console.print(Text(line, style=_CORAL))
        self._logo_done = True

    def _print_close(self) -> None:
        elapsed = self._section_timer.stop()
        section_attrs = self._messages[-1].attrs if self._messages else {}
        section_details = self.create_details(section_attrs)

        if self._failed:
            mark = "✗"
            style = _RED
        else:
            mark = "✔"
            style = _GREEN

        head = Text()
        head.append(f" {mark} ", style=style)
        head.append(self._task_label(), style=style)
        head.append(
            f" ({self._format_inline(elapsed, section_details)})", style=_GREY)
        self._console.print(head)

        if not self._failed:
            return

        for m in self._messages:
            details = self.create_details(m.attrs)
            inline = self._format_inline(m.elapsed, details)

            if m.level == "error":
                row_style = _RED
            elif m.level == "warning":
                row_style = _YELLOW
            else:
                row_style = _GREY

            line = Text()
            line.append(_INDENT)
            line.append(m.text, style=row_style)
            line.append(f" ({inline})", style=_GREY)
            self._console.print(line)

            if m.traceback:
                self._console.print(Text(m.traceback, style=_RED))

    def __call__(self, logger: Any, method: str, event_mapping: MutableMapping[str, Any]) -> Any:
        event_dict = dict(event_mapping)
        message = event_dict.pop("event", "")
        level = event_dict.pop("level", "")
        event_dict.pop("timestamp", None)

        tb_text = traceback.format_exc() if level == "error" else None

        new_task_name = event_dict.pop("task_name", _UNSET)
        is_end = new_task_name is END_SECTION
        section_change = (
            not is_end
            and new_task_name is not _UNSET
            and new_task_name != self._task_name
        )

        with self._lock:
            if not self._logo_done:
                self._print_logo()

            if not self._live_started:
                self._live.start()
                self._live_started = True

            now = time.monotonic()

            if self._has_active:
                self._messages[-1].elapsed = now - self._message_start

            if is_end:
                if self._has_active:
                    self._has_active = False
                    self._print_close()
                self._task_name = _UNSET
                self._messages = []
                self._failed = False
                if self._live_started:
                    self._live.stop()
                    self._live_started = False
                raise structlog.DropEvent()

            if section_change:
                if self._has_active:
                    self._print_close()
                self._task_name = new_task_name
                self._messages = []
                self._failed = False
                self._loader = get_random_loader()
                self._section_timer.start()
                self._has_active = False

            if level == "error":
                self._failed = True

            self._messages.append(_Message(
                text=message,
                level=level,
                attrs=event_dict,
                elapsed=0.0,
                traceback=tb_text,
            ))
            self._message_start = now
            self._has_active = True

        raise structlog.DropEvent()
