import logging
import sys
import os
import traceback
from datetime import datetime

# Custom level for blank separator lines (no timestamp, no prefix)
BLANK_LEVEL = 21
logging.addLevelName(BLANK_LEVEL, "BLANK")


_blank_in_progress = False


def _logging_blank(n=1, **kwargs):
    """Emit N blank lines with no timestamp."""
    global _blank_in_progress
    if _blank_in_progress:
        return
    _blank_in_progress = True
    try:
        logging.log(BLANK_LEVEL, str(n))
    finally:
        _blank_in_progress = False


logging.blank = _logging_blank


class TracebackHandler(logging.Handler):
    def __init__(self, wrapped_handler, listHandler):
        super().__init__()
        self.wrapped_handler = wrapped_handler
        self.setFormatter(wrapped_handler.formatter)
        self.setLevel(wrapped_handler.level)
        self.listHandler = listHandler

    def _emit_blank(self, n=1):
        """Emit blank lines directly to the wrapped handler only."""
        blank_record = logging.LogRecord(
            name='root', level=BLANK_LEVEL, pathname='', lineno=0,
            msg=str(n), args=(), exc_info=None
        )
        blank_record.levelname = 'BLANK'
        self.wrapped_handler.emit(blank_record)

    def emit(self, record):
        if record.exc_info or record.stack_info:
            # Handle based on log level
            # For CRITICAL level: do full processing then exit
            # Get user code stack (filter system/logging internals)
            current_stack = traceback.format_stack()[:-4]  # Remove logging frames
            user_stack = False
            clean_exception = False
            try:
                tb = traceback.format_exception(*record.exc_info)
            except Exception:
                tb = traceback.format_exception(*record.stack_info)
            if record.levelno in (logging.CRITICAL, logging.INFO, logging.ERROR):
                user_stack = self._filter_user_code(current_stack) or []

                # Get clean exception traceback
                clean_exception = self._filter_user_code(tb) or []
            elif record.levelno == logging.WARNING:
                user_stack = []
                clean_exception = self._filter_user_code(tb) or []

            # Prepare combined trace for formatter
            # Avoid duplicating entries that appear both in the current
            # call path and in the exception traceback.
            if user_stack:
                stack_section = "Call Path:\n" + '\n'.join(user_stack)
            else:
                stack_section = ""

            if clean_exception:
                # Keep only exception details that are not already present
                # in the call path
                exception_only = [
                    line for line in clean_exception if line not in user_stack
                ]
                if exception_only:
                    exception_section = (
                        "Error Details:\n"
                        + ''.join(exception_only)
                    )
                else:
                    # If nothing new in the exception after deduping,
                    # leave empty
                    exception_section = ""
            else:
                # Fallback: include raw exception text if filtering failed
                exception_section = ''.join(tb)  # Fallback to full trace
            
            # Store combined trace in record for formatter to use
            if stack_section:
                record.combined_trace = f"{stack_section}\n{exception_section}"
            else:
                record.combined_trace = exception_section

            # Emit the log with blank lines around it for ERROR/CRITICAL
            if record.levelno in (logging.ERROR, logging.CRITICAL):
                self._emit_blank(n=2)
            self.wrapped_handler.emit(record)
            if record.levelno in (logging.ERROR, logging.CRITICAL):
                self._emit_blank(n=2)

            # Exit on CRITICAL level
            # Exit if: console handler OR file handler with no console
            if record.levelno == logging.CRITICAL:
                handler_name = getattr(self.wrapped_handler, 'name', '')
                if handler_name == 'ConsoleHandler':
                    sys.exit(1)
                elif (handler_name == 'FileHandler' and
                      'ConsoleHandler' not in self.listHandler):
                    sys.exit(1)
        else:
            # No exception info, emit normally
            import re as _re
            is_step = bool(_re.search(r'=== Step \d+', record.getMessage()))
            if record.levelno in (logging.ERROR, logging.CRITICAL):
                self._emit_blank(n=2)
            elif is_step:
                self._emit_blank(n=4)
            self.wrapped_handler.emit(record)
            if record.levelno in (logging.ERROR, logging.CRITICAL):
                self._emit_blank(n=2)
            elif is_step:
                self._emit_blank(n=1)
    
    def _filter_user_code(self, stack_lines):
        """
        Filter out system and logging internal frames.
        Keep only user code frames.
        """
        filtered = []
        skip_patterns = [
            'logging\\__init__.py',
            'logging\\handlers.py',
            'WindowsApps\\Python',
            'site-packages',
            'lib\\python',
            'concurrent\\futures',
            'threading.py'
        ]
        
        for line in stack_lines:
            # Keep user code frames
            should_skip = any(pattern in line for pattern in skip_patterns)
            if not should_skip and ('File "' in line):
                # Make file paths clickable by removing quotes and
                # showing full path
                if 'File "' in line:
                    # Extract and format the file path for VS Code
                    # clickability
                    import re
                    pattern = r'File "([^"]+)", line (\d+), in (.+)'
                    match = re.search(pattern, line)
                    if match:
                        file_path, line_num, func_name = match.groups()
                        # Keep full path for clickability, format nicely
                        formatted_line = (
                            "  File " + file_path + ":" + line_num
                            + " in " + func_name
                        )
                        filtered.append(formatted_line)
                else:
                    filtered.append(line)
            elif not line.strip().startswith('File "') and filtered:
                # Keep code lines that follow user files, add spacing
                filtered.append(f"    {line.strip()}")
                # Add spacing between file entries
                if line.strip():
                    filtered.append("")
        
        return filtered
        

class ColoredFormatter(logging.Formatter):
    """Custom formatter with bright ANSI colors for console output."""
    COLORS = {
        'DEBUG': '\033[96m',      # Bright cyan
        'INFO': '\033[92m',       # Bright green
        'WARNING': '\033[38;2;255;220;0m',  # Vivid yellow (RGB)
        'ERROR': '\033[91m',      # Bright red
        'CRITICAL': '\033[95m',   # Bright magenta
        'RESET': '\033[0m',       # Reset color
        'TIMESTAMP': '\033[38;2;34;139;34m',  # Forest green (low saturation)
        'STEP': '\033[38;2;100;134;219m'      # Blue (#6486db) for === Step N ===
    }
    
    def format(self, record):
        """Format log record with bright colors and timestamp."""
        if record.levelname == "BLANK":
            try:
                n = max(1, int(record.getMessage())) - 1
            except (ValueError, TypeError):
                n = 0
            return "\n" * n
        # Format timestamp with forest green color - more readable format
        timestamp_str = datetime.fromtimestamp(record.created).strftime(
            '%Y-%m-%d %H:%M:%S'
        )
        colored_timestamp = (
            f"{self.COLORS['TIMESTAMP']}{timestamp_str}{self.COLORS['RESET']}"
        )
        # Format log level with bright color
        level_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        colored_level = (
            f"{level_color}{record.levelname}{self.COLORS['RESET']}"
        )
        # Colorize === Step N: ... === patterns in the message
        import re as _re
        raw_msg = record.getMessage()
        raw_msg = _re.sub(
            r'(=== Step \d+.*?===)',
            lambda m: (
                f"{self.COLORS['STEP']}{m.group(1)}{self.COLORS['RESET']}"
            ),
            raw_msg
        )
        # Build complete formatted message
        formatted_msg = (
            f"{colored_timestamp} [{colored_level}] : "
            f"{record.name} : {raw_msg}"
        )

        # Add combined trace information if present
        if hasattr(record, 'combined_trace') and record.combined_trace:
            # Use the pre-processed clean trace from TracebackHandler
            full_traceback = record.combined_trace
            traceback_lines = full_traceback.split('\n')
            
            # Process lines with spacing between file entries
            formatted_lines = []
            previous_was_file = False
            
            for line in traceback_lines:
                if line.strip():
                    # Add spacing before new file entries
                    # (except the first one)
                    if (
                        line.strip().startswith('  File ')
                        and previous_was_file
                    ):
                        formatted_lines.append('│')

                    formatted_lines.append(line)
                    previous_was_file = line.strip().startswith('  File ')
            
            # Calculate max width including separators
            # First compute a preliminary width based on the raw lines
            all_lines = formatted_lines
            # Limit width to terminal size to avoid extremely wide boxes
            try:
                import shutil
                size = shutil.get_terminal_size(fallback=(160, 50))
                term_width = size.columns
            except Exception:
                term_width = 120
                
            term_cap = int(int(term_width) * 0.95)
            # Wrap lines that are longer than the cap, preserving leading
            # whitespace
            try:
                import textwrap
                import re
            except Exception:
                textwrap = None
                re = None

            prelim_max = max(
                (len(line) for line in all_lines if line != '│'),
                default=70,
            )
            max_width = min(prelim_max, term_cap)

            # If any lines exceed max_width, wrap them into multiple
            # lines
            if textwrap and re:
                wrapped_lines = []
                for line in all_lines:
                    if line == '│':
                        wrapped_lines.append(line)
                        continue
                    leading = re.match(r'^(\s*)', line).group(1)
                    available = max_width - len(leading)
                    if available <= 0:
                        available = max_width
                    parts = (
                        textwrap.wrap(line.strip(), width=available)
                        or [line.strip()]
                    )
                    for part in parts:
                        wrapped_lines.append(f"{leading}{part}")
                all_lines = wrapped_lines
                # recompute max_width from wrapped lines to avoid extra
                # padding
                max_width = min(
                    max((len(line) for line in all_lines if line != '│'),
                        default=50),
                    term_cap,
                )
            # Box width = 2 chars for borders + max_width + 2 chars for spaces
            box_width = max_width + 4
            
            # Format each line with proper spacing
            # Build formatted traceback lines with colors and borders
            ft_lines = []
            for line in all_lines:
                if line == '│':
                    blank = ' ' * (max_width + 2)
                    ft_lines.append(
                        f"{self.COLORS['ERROR']}│{self.COLORS['RESET']}{blank}"
                        f"{self.COLORS['ERROR']}│{self.COLORS['RESET']}"
                    )
                else:
                    ft_lines.append(
                        f"{self.COLORS['ERROR']}│{self.COLORS['RESET']} "
                        f"{line:<{max_width}} "
                        f"{self.COLORS['ERROR']}│{self.COLORS['RESET']}"
                    )
            formatted_traceback = '\n'.join(ft_lines)
            
            # Top: ┌─ TRACEBACK (11 chars) + remaining dashes + ┐ = box_width
            top_dashes = box_width - 12  # 12 = len("┌─ TRACEBACK ┐")
            # Build boxed header/footer in smaller parts to avoid long lines
            header = (
                f"{self.COLORS['ERROR']}┌─ TRACEBACK "
                + ('─' * (top_dashes - 2))
                + f"┐{self.COLORS['RESET']}\n"
            )
            footer = (
                f"{self.COLORS['ERROR']}└" + ('─' * (box_width - 2))
                + f"┘{self.COLORS['RESET']}"
            )
            formatted_msg += (
                "\n"
                + header
                + formatted_traceback
                + "\n"
                + footer
            )
        elif hasattr(record, 'full_stack') and record.full_stack:
            # Display full stack trace when no exception but stack was captured
            stack_lines = []
            stack_lines.append("Traceback (most recent call last):")
            for frame in record.full_stack:
                file_line = (
                    '  File "{}", line {}, in {}'.format(
                        frame.filename, frame.lineno, frame.name
                    )
                )
                stack_lines.append(file_line)
                if frame.line:
                    stack_lines.append(f'    {frame.line}')
            
            # Filter out empty lines and calculate max width
            clean_lines = [line for line in stack_lines if line.strip()]
            # Limit width to terminal size to avoid extremely wide boxes
            try:
                import shutil
                size = shutil.get_terminal_size(fallback=(120, 25))
                term_width = size.columns
            except Exception:
                term_width = 120
            # Reduce max width to 70% of terminal width to avoid overflow
            term_cap = max(int(term_width * 0.7), 40)

            try:
                import textwrap
                import re
            except Exception:
                textwrap = None
                re = None

            prelim_max = max((len(line) for line in clean_lines), default=50)
            max_width = min(prelim_max, term_cap)

            if textwrap and re:
                wrapped_clean = []
                for line in clean_lines:
                    leading = re.match(r'^(\s*)', line).group(1)
                    available = max_width - len(leading)
                    if available <= 0:
                        available = max_width
                    parts = (
                        textwrap.wrap(line.strip(), width=available)
                        or [line.strip()]
                    )
                    for part in parts:
                        wrapped_clean.append(f"{leading}{part}")
                clean_lines = wrapped_clean
                max_width = min(
                    max((len(line) for line in clean_lines), default=50),
                    term_cap,
                )
            # Box width = 2 chars for borders + max_width + 2 chars for spaces
            box_width = max_width + 4
            
            # Build formatted traceback for plain formatter
            formatted_traceback = '\n'.join(
                [f"│ {line:<{max_width}} │" for line in clean_lines]
            )
            # Top: ┌─ STACK TRACE (13 chars) + remaining dashes + ┐ = box_width
            top_dashes = box_width - 14  # 14 = len("┌─ STACK TRACE ┐")
            # Build boxed stack trace header/footer in smaller parts
            st_header = (
                f"{self.COLORS['ERROR']}┌─ STACK TRACE "
                + ('─' * (top_dashes - 2))
                + f"┐{self.COLORS['RESET']}\n"
            )
            st_footer = (
                f"{self.COLORS['ERROR']}└" + ('─' * (box_width - 2))
                + f"┘{self.COLORS['RESET']}"
            )
            formatted_msg += (
                "\n"
                + st_header
                + formatted_traceback
                + "\n"
                + st_footer
            )
        
        return formatted_msg


class PlainFormatter(logging.Formatter):
    """Plain formatter without colors for file output."""
    def format(self, record):
        """Format log record without colors for clean file output."""
        if record.levelname == "BLANK":
            try:
                n = max(1, int(record.getMessage())) - 1
            except (ValueError, TypeError):
                n = 0
            return "\n" * n
        # Format timestamp without colors - more readable format
        timestamp_str = datetime.fromtimestamp(record.created).strftime(
            '%Y-%m-%d %H:%M:%S'
        )
        # Build clean formatted message
        formatted_msg = (
            f"{timestamp_str} [{record.levelname}] : "
            f"{record.name} : {record.getMessage()}"
        )
        
        # Add combined trace information if present
        if hasattr(record, 'combined_trace') and record.combined_trace:
            # Use the pre-processed clean trace from TracebackHandler
            full_traceback = record.combined_trace
            traceback_lines = full_traceback.split('\n')
            # Filter out empty lines and calculate max width
            clean_lines = [line for line in traceback_lines if line.strip()]
            max_width = (
                max(len(line) for line in clean_lines)
                if clean_lines
                else 50
            )
            # Box width = 2 chars for borders + max_width + 2 chars for spaces
            box_width = max_width + 4
            
            formatted_traceback = '\n'.join([
                f"│ {line:<{max_width}} │" for line in clean_lines
            ])
            # Top: ┌─ TRACEBACK (11 chars) + remaining dashes + ┐ = box_width
            top_dashes = box_width - 12  # 12 = len("┌─ TRACEBACK ┐")
            formatted_msg += (
                f"\n┌─ TRACEBACK {'─' * (top_dashes -2)}┐\n"
                f"{formatted_traceback}\n"
                f"└{'─' * (box_width - 2)}┘"
            )
        elif hasattr(record, 'full_stack') and record.full_stack:
            # Display full stack trace when no exception but stack was captured
            stack_lines = []
            stack_lines.append("Traceback (most recent call last):")
            for frame in record.full_stack:
                file_line = (
                    '  File "{}", line {}, in {}'.format(
                        frame.filename, frame.lineno, frame.name
                    )
                )
                stack_lines.append(file_line)
                if frame.line:
                    stack_lines.append(f'    {frame.line}')
            
            # Filter out empty lines and calculate max width
            clean_lines = [line for line in stack_lines if line.strip()]
            max_width = (
                max(len(line) for line in clean_lines)
                if clean_lines
                else 50
            )
            # Box width = 2 chars for borders + max_width + 2 chars for spaces
            box_width = max_width + 4
            
            formatted_traceback = '\n'.join([
                f"│ {line:<{max_width}} │" for line in clean_lines
            ])
            # Top: ┌─ STACK TRACE (13 chars) + remaining dashes + ┐ = box_width
            top_dashes = box_width - 14  # 14 = len("┌─ STACK TRACE ┐")
            formatted_msg += (
                f"\n┌─ STACK TRACE {'─' * (top_dashes -2)}┐\n"
                f"{formatted_traceback}\n"
                f"└{'─' * (box_width - 2)}┘"
            )
        
        return formatted_msg


def setup_logging(
    level=logging.INFO,
    log_file=None,
    console=True,
    cleanup=True,
    enable_traceback=True
):
    """
    Setup logging with optional console and file output.
    Automatically keeps only root logger at specified level, others at WARNING.
    Args:
        level: Logging level for root logger (default: INFO)
        log_file: Optional log file path. If "log_file", creates timestamped
            file in logs/ directory (clean text without colors)
        console: Whether to display logs in console (default: True)
        cleanup: Whether to clean the log file before starting (default: True)
        enable_traceback:
            Whether to automatically add traceback to error/critical logs
    """
    # Handle automatic log file creation
    _skip_cleanup = False  # prevents wiping a reused or freshly created file

    if log_file == "log_file":
        # Auto-managed files are never wiped (marker line must be preserved)
        _skip_cleanup = True

        # Create logs directory at the same level as this file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(script_dir, "logs")
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        # The PPID is shared by all Python scripts launched by the same shell
        # script (run_ingest.sh).  We embed it as the very first line of the
        # log file so sibling processes can rediscover the file without any
        # external pointer file.
        ppid = os.getppid()
        ppid_marker = f"# TESTPILOT_RUN_PPID={ppid}"

        _reused = False
        try:
            # Scan existing log files, newest first, for one that belongs to
            # this execution (same PPID written on line 1).
            _candidates = sorted(
                (
                    f for f in os.listdir(logs_dir)
                    if "loginfo" in f.lower() and f.lower().endswith(".log")
                ),
                reverse=True,
            )
            for _fname in _candidates:
                _fpath = os.path.join(logs_dir, _fname)
                try:
                    with open(_fpath, "r", encoding="utf-8") as _f:
                        _first_line = _f.readline().strip()
                    if _first_line == ppid_marker:
                        log_file = _fpath
                        _reused = True
                        break
                except OSError:
                    pass
        except OSError:
            pass

        if not _reused:
            # Clean up old log files 
            cleanup_old_log_files(logs_dir, max_files=3)

            # Generate timestamp for log filename with better readability
            timestamp = datetime.now().strftime("loginfo_%Y-%m-%d_%H%M%S")
            log_file = os.path.join(logs_dir, f"{timestamp}.log")

            # Write the PPID marker as the first line so sibling processes
            # can find and reuse this file.
            try:
                with open(log_file, "w", encoding="utf-8") as _f:
                    _f.write(ppid_marker + "\n")
            except OSError:
                pass

        # Also propagate via env var for genuine child processes
        os.environ["TESTPILOT_LOG_FILE"] = log_file
    
    # Set custom logger class for automatic traceback
    if enable_traceback:
        pass  # We'll use handler wrapping instead
    
    # Clean up log file first if specified (never for auto-managed files)
    if log_file and cleanup and not _skip_cleanup:
        cleanup_log_file(log_file)
    
    # Clear existing handlers to avoid duplicates
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # File handler with clean text
    Handlers = []
    if log_file:
        Handlers.append('FileHandler')
        file_handler = logging.FileHandler(
            log_file, mode='a', encoding='utf-8'
        )
        file_handler.name = 'FileHandler'
        file_formatter = PlainFormatter()
        file_handler.setFormatter(file_formatter)
        
        # Wrap with auto-traceback if enabled
        if enable_traceback:
            file_handler = TracebackHandler(
                file_handler, listHandler=Handlers
            )
            
        logging.root.addHandler(file_handler)
    # Console handler with bright colors (if enabled)
    if console:
        Handlers.append('ConsoleHandler')
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.name = 'ConsoleHandler'
        console_formatter = ColoredFormatter()
        console_handler.setFormatter(console_formatter)
        
        # Wrap with auto-traceback if enabled
        if enable_traceback:
            console_handler = TracebackHandler(
                console_handler, listHandler=Handlers
            )
            
        logging.root.addHandler(console_handler)

    # Set root logger level
    logging.root.setLevel(level)

    # Set higher threshold for all non-root loggers
    class RootOnlyFilter(logging.Filter):
        def filter(self, record):
            # Allow all levels from root logger and pipeline namespace
            if record.name == 'root':
                return True
            if record.name.startswith('pipeline'):
                return True
            return record.levelno >= logging.WARNING
    
    # Apply filter to all handlers
    for handler in logging.root.handlers:
        handler.addFilter(RootOnlyFilter())


def cleanup_old_log_files(logs_dir, max_files=3):
    """
    Clean up old log files, keeping only the most recent ones.
    
    Args:
        logs_dir (str): Directory containing log files
        max_files (int): Maximum number of log files to keep (default: 5)
    """
    try:
        # Check if directory exists first
        if not os.path.exists(logs_dir):
            return
        
        # Get all testpilot log files
        log_files = []
        for filename in os.listdir(logs_dir):
            name_lower = filename.lower()
            if ("loginfo" in name_lower) and name_lower.endswith(".log"):
                file_path = os.path.join(logs_dir, filename)
                # Get file modification time for sorting
                mtime = os.path.getmtime(file_path)
                log_files.append((mtime, file_path, filename))
        # Sort by modification time (newest first)
        log_files.sort(reverse=True)
        
        # Remove old files if we have more than max_files
        if len(log_files) >= max_files:
            # Keep newest max_files-1 entries, remove the rest
            files_to_remove = log_files[max_files - 1:]
            for _, file_path, filename in files_to_remove:
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(
                        "Warning: Could not remove old log file "
                        + filename
                        + ": "
                        + str(e)
                    )
                    
    except Exception as e:
        print(f"Warning: Error during log file cleanup: {e}")


def cleanup_log_file(log_file_path):
    """
    Clean up (empty) the specified log file.
    
    Args:
        log_file_path (str): Path to the log file to clean up
        
    Raises:
        FileNotFoundError: If the log file does not exist
        Exception: For other file system errors
    """
    try:
        with open(log_file_path, 'w', encoding='utf-8') as f:
            f.write('')
    except FileNotFoundError:
        print(f"Warning: Log file not found: {log_file_path}")
    except Exception as e:
        print(f"Error cleaning log file {log_file_path}: {e}")