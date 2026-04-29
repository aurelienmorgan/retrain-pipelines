
import os
import sys
import logging
import inspect
import builtins
import warnings

from typing import Dict, TextIO
from datetime import datetime

from contextlib import contextmanager, \
    redirect_stdout, redirect_stderr

from rich.text import Text
from rich.console import Console
from rich.highlighter import ReprHighlighter

from ..utils import in_notebook
if in_notebook():
    from .nb_console_print import \
        nb_activate, nb_deactivate, \
        nb_console_print


# Module-level controller instance for picklable print wrapper
_global_controller = None


def _patched_print(*args, **kwargs):
    """Module-level print wrapper for cloudpickle compatibility."""
    if _global_controller is not None and _global_controller._active:
        return _global_controller._custom_print(*args, **kwargs)
    else:
        # Fallback to original print
        orig = _global_controller._orig_print if _global_controller \
               else builtins.__dict__['print']
        return orig(*args, **kwargs)


def _patched_stdout_write(s):
    """Module-level stdout.write wrapper for cloudpickle compatibility."""
    if _global_controller is not None and _global_controller._active:
        return _global_controller._custom_write(s, False)
    else:
        orig = _global_controller._orig_stdout_write if _global_controller \
               else sys.__stdout__.write
        return orig(s)


def _patched_stderr_write(s):
    """Module-level stderr.write wrapper for cloudpickle compatibility."""
    if _global_controller is not None and _global_controller._active:
        return _global_controller._custom_write(s, True)
    else:
        orig = _global_controller._orig_stderr_write if _global_controller \
               else sys.__stderr__.write
        return orig(s)



class CustomRichHandler(logging.Handler):
    """
    Rich handler that ALWAYS writes to CURRENT sys.stdout/sys.stderr.
    This guarantees compatibility with StreamToDb redirection.
    """

    LEVEL_STYLES = {
        "DEBUG": "bright_cyan",
        "INFO": "",
        "WARNING": "yellow",
        "ERROR": "bold red",
        "CRITICAL": "bold white on red",
    }

    def __init__(self, markup: bool = True, level=logging.NOTSET):
        super().__init__(level=level)
        self.markup = markup
        self.highlighter = ReprHighlighter()

    def emit(self, record: logging.LogRecord):
        try:
            # IMPORTANT:
            # Console must bind to sys.stdout/sys.stderr AT EMIT TIME
            console = Console(
                file=sys.stderr if record.levelno >= logging.WARNING else sys.stdout,
                soft_wrap=False,
                width=1_000_000,
                force_terminal=True,
                legacy_windows=False,
            )

            ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
            timestamp = Text(f"[{ts}] ", style="bright_black")

            level_text = Text()
            if record.levelname != "INFO":
                style = self.LEVEL_STYLES.get(record.levelname, "")
                level_text = Text(f"{record.levelname}: ", style=style)

            filename = os.path.basename(record.pathname)
            path = Text.assemble((
                f"{filename}:{record.lineno}",
                f"link file://{record.pathname}"
            ))
            path.append(
                Text(" ")
                + Text(
                    f" {record.name} {os.getpid()} ",
                    style="italic rgb(78,78,78) on rgb(26,26,26)",
                )
                + Text(" ")
            )
            path.append("\n")

            msg_str = record.getMessage()
            body = (
                Text.from_markup(msg_str)
                if self.markup
                else Text(msg_str)
            )
            body = self.highlighter(body)

            rich_formatted_message = Text.assemble(
                # Text(str(console.file.__class__.__name__)),
                timestamp, level_text, path, body
            )

            ########################################
            # console.print(rich_formatted_message, sep="")
            ########################################
            from rich.console import Capture
            with Capture(console) as capture:
                console.print(rich_formatted_message, sep="")

            ansi_output = capture.get()
            # print(repr(ansi_output))  # '\x1b[1ma message in bold\x1b[0m'
            from retrain_pipelines.dag_engine.core.core import StreamToDb
            if not in_notebook() or isinstance(console.file, StreamToDb):
                console.file.write(ansi_output)     # explicit call
                                                    # to StreamToDb.write
                                                    # for task-traces contexts
                                                    # when execution runs
                                                    # in a Notebook cell
            if in_notebook():
                nb_console_print(console, rich_formatted_message)
            console.file.flush()

        except Exception:
            self.handleError(record)


class RichLoggingController:
    """Controller that activates/deactivates Rich logging and I/O patches."""

    def __init__(self):
        self._active = False

        self._orig_getLogger = logging.getLogger
        self._orig_print = builtins.print
        self._orig_stdout_write = sys.stdout.write
        self._orig_stderr_write = sys.stderr.write
        self._orig_root_level = logging.getLogger().level

        warnings_logger = logging.getLogger("py.warnings")
        self._warnings_handler_backup = list(warnings_logger.handlers)
        self._warnings_level_backup = warnings_logger.level
        self._warnings_propagate_backup = warnings_logger.propagate

        self._logger_to_rich: Dict[str, CustomRichHandler] = {}

    def __getstate__(self):
        """Prepare controller for pickling - mark as inactive."""
        builtins.print = self._orig_print
        return {'_active': False}

    def __setstate__(self, state):
        """Restore controller after unpickling - reinitialize in subprocess."""
        self.__init__()
        builtins.print = _patched_print
        self._active = False

    def _getLogger_wrapped(self, name: str = None):
        logger = self._orig_getLogger(name)

        if name in self._logger_to_rich:
            return logger

        package_name = __name__.split(".")[0]

        if (
            (not name or package_name == name) and
            not any(
                isinstance(h, CustomRichHandler)
                for h in logger.handlers
            )
        ):
            logger.handlers += [CustomRichHandler(markup=True)]

        if (
            not (name and name.startswith("uvicorn"))
            and isinstance(logger, logging.Logger)
        ):
            logger.handlers = [
                CustomRichHandler(markup=True, level=h.level)
                if (
                    isinstance(h, logging.StreamHandler)
                    and getattr(h, "stream", None)
                        in {sys.stdout, sys.stderr}
                ) else h
                for h in logger.handlers
            ]

            self._logger_to_rich[name] = logger

        return logger

    def activate(self):
        """Install patches (idempotent)."""
        global _global_controller

        if self._active:
            return

        _global_controller = self

        if in_notebook():
            nb_activate(self._orig_stdout_write)

        logging.getLogger = self._getLogger_wrapped

        package_name = __name__.split(".")[0]
        logging.getLogger(package_name).setLevel(logging.DEBUG)
        logging.getLogger(package_name).propagate = False
        logging.getLogger().setLevel(logging.INFO)

        for name, logger in logging.root.manager.loggerDict.items():
            if name == "rich":
                continue
            if isinstance(logger, logging.Logger):
                logging.getLogger(logger.name)

        warnings_logger = logging.getLogger("py.warnings")
        warnings_logger.handlers = []
        warnings_logger.addHandler(CustomRichHandler(markup=True))
        warnings_logger.propagate = False
        warnings_logger.setLevel(logging.WARNING)

        builtins.print = _patched_print
        sys.stdout.write = _patched_stdout_write
        sys.stderr.write = _patched_stderr_write

        self._active = True

    def deactivate(self):
        """Remove patches and restore originals (idempotent)."""
        global _global_controller

        if not self._active:
            return

        if in_notebook():
            nb_deactivate()

        logging.getLogger = self._orig_getLogger
        logging.getLogger().setLevel(self._orig_root_level)

        builtins.print = self._orig_print
        sys.stdout.write = self._orig_stdout_write
        sys.stderr.write = self._orig_stderr_write

        warnings_logger = logging.getLogger("py.warnings")
        warnings_logger.handlers = self._warnings_handler_backup
        warnings_logger.level = self._warnings_level_backup
        warnings_logger.propagate = self._warnings_propagate_backup

        # remove all CustomRichHandler instances
        # from tracked loggers
        for logger_name in list(self._logger_to_rich.keys()):
            logger = logging.getLogger(logger_name)
            logger.handlers = [
                h for h in logger.handlers 
                if not isinstance(h, CustomRichHandler)
            ]
        
        self._logger_to_rich.clear()

        _global_controller = None
        self._active = False

    def _rich_emit(
        self,
        source_name,
        args_or_text=None,
        is_err=False,
        frame_offset=2,
        **kwargs,
    ):
        """
        Unified Rich emit for print/write.
        ALWAYS writes to CURRENT sys.stdout/sys.stderr.
        """

        from rich.console import Console
        from rich.text import Text
        from rich.highlighter import ReprHighlighter
        import inspect
        from datetime import datetime
        import os
        import sys

        stream = sys.stderr if is_err else sys.stdout

        console = Console(
            file=stream,
            soft_wrap=False,
            width=1_000_000,
            force_terminal=True,
            legacy_windows=False,
        )

        highlighter = ReprHighlighter()

        frame = inspect.currentframe()
        for _ in range(frame_offset):
            frame = frame.f_back

        filename = os.path.basename(frame.f_code.co_filename)
        full_path = frame.f_code.co_filename
        lineno = frame.f_lineno

        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        timestamp = Text(f"[{ts}] ", style="bright_black")

        path = Text.assemble(
            (f"{filename}:{lineno}", f"link file://{full_path}")
        )
        path.append(
            Text(" ")
            + Text(
                f" {source_name} {os.getpid()} ",
                style="italic rgb(78,78,78) on rgb(26,26,26)",
            )
            + Text(" ")
        )
        path.append("\n")

        if isinstance(args_or_text, str):
            msg = args_or_text.rstrip("\n")
        else:
            msg = " ".join(str(a) for a in args_or_text)

        ########################################
        body = Text.from_markup(msg)
        ########################################
        #      DEBUG STACK       -  BEGIN      #
        ########################################
        # stack = inspect.stack()
        # frames_info = []
        # for frame_info in stack: # [frame_offset:]:
            # fname = os.path.basename(frame_info.filename)
            # lineno = frame_info.lineno
            # frames_info.append(f"{fname}:{lineno}")

        # all_frames_str = "\n".join(frames_info)
        # body = Text.from_markup(msg + "\n" + all_frames_str)
        ########################################
        #      DEBUG STACK       -   END       #
        ########################################
        body = highlighter(body)

        end = kwargs.get("end", "\n")

        rich_formatted_message = Text.assemble(
            #Text(str(console.file.__class__.__name__)),
            timestamp, path, body
        )

        ########################################
        # console.print(rich_formatted_message, end=end)
        ########################################
        from rich.console import Capture
        with Capture(console) as capture:
            console.print(rich_formatted_message, sep="")

        ansi_output = capture.get()
        # print(repr(ansi_output))  # '\x1b[1ma message in bold\x1b[0m'
        from retrain_pipelines.dag_engine.core.core import StreamToDb
        if not in_notebook() or isinstance(console.file, StreamToDb):
            console.file.write(ansi_output)     # explicit call
                                                # to StreamToDb.write
                                                # for task-traces contexts
                                                # when execution runs
                                                # in a Notebook cell
        if in_notebook():
            nb_console_print(console, rich_formatted_message)
        console.file.flush()


    def _custom_print(self, *args, **kwargs):
        from retrain_pipelines.dag_engine.core.core import StreamToDb

        if (
            "file" in kwargs
            and kwargs["file"] is not None
            and not isinstance(kwargs["file"], StreamToDb)
        ):
            return self._orig_print(*args, **kwargs)

        is_err = (
            "file" in kwargs
            and isinstance(kwargs["file"], StreamToDb)
            and kwargs["file"].is_err
        )

        self._rich_emit("print", args, is_err, **kwargs,
                        frame_offset=3)#(2 if is_err else 3))

    def _custom_write(self, s, is_err=False):
        original = self._orig_stderr_write if is_err \
                   else self._orig_stdout_write

        if not str(s).strip():
            return original(s)

        frame = inspect.currentframe()

        if frame:
            try:
                frame = frame.f_back
                while frame is not None:
                    module = frame.f_globals.get("__name__", "")
                    func_name = frame.f_code.co_name

                    if (
                        module.startswith("code")
                        or module.startswith("codeop")
                        or func_name == "excepthook"
                    ):
                        return original(s)

                    if (
                        module.startswith("rich.console")
                        or module.startswith("rich.live")
                        or func_name in ("_rich_emit", "emit")
                    ):
                        return None if in_notebook() else original(s)

                    frame = frame.f_back
            except Exception:
                return original(s)
            finally:
                del frame

        try:
            self._rich_emit(
                "write", s, is_err,
                frame_offset=(4+(2 if in_notebook() else 0))
            ) # frame_offset 4
              # in notebook: 6, due to "write" specific routing
            return len(s)
        except Exception:
            return original(s)


@contextmanager
def rp_redirect_stdout(file: TextIO):
    """Context manager to redirect streamhandlers

    when RichLoggingController is active.
    
    Temporarily deactivates the controller
    and adds a basic StreamHandler AFTER
    redirecting, so it writes to the redirected stream.
    
    Params:
        - file (TextIO):
            File object to redirect stdout to
        
    Example:
        with open('output.txt', 'w') as f:
            with rp_redirect_stdout(f):
                print("This goes to the file")
                logging.warning("This too, plain format")
    """
    if _global_controller is None or not _global_controller._active:
        # Controller not active, use normal redirect
        with redirect_stdout(file), redirect_stderr(file):
            yield
    else:
        controller = _global_controller
        controller.deactivate()
        
        try:
            with redirect_stdout(file), redirect_stderr(file):
                # Create handler AFTER redirect
                # so sys.stdout is already the file
                root_logger = logging.getLogger()
                # make sys.stdout BE the file
                temp_handler = logging.StreamHandler(sys.stdout)
                temp_handler.setFormatter(logging.Formatter(
                    '%(levelname)s:%(name)s:%(message)s'))
                root_logger.addHandler(temp_handler)
                
                try:
                    yield
                finally:
                    # Clean up temp handler
                    root_logger.removeHandler(temp_handler)
        finally:
            # Reactivate controller
            controller.activate()

