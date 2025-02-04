import sys
import os
import atexit
from datetime import datetime

class Tee:
    """A class that duplicates output to multiple streams (e.g., console + file)."""
    
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            try:
                stream.write(message)
                stream.flush()  # Ensure immediate writing
            except (OSError, ValueError):
                # Ignore errors when writing to closed streams
                pass

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except (OSError, ValueError):
                pass

def setup_logging(log_dir="log"):
    """
    Redirects stdout and stderr to both console and a log file in a specific folder.

    Args:
        log_dir (str): Folder where logs will be saved.

    Returns:
        log_file (file object): The opened log file (must be closed later).
    """
    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Generate a timestamped log filename inside the specified folder
    log_filename = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt")
    log_file = open(log_filename, "w")  # Open the log file

    # Redirect stdout (print) and stderr (errors) to both console and file
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    # Register cleanup function to close log file when the script exits
    atexit.register(lambda: safe_close(log_file))

    return log_file  # Return the log file in case explicit closing is needed

def safe_close(file):
    """Safely close the log file to avoid issues on exit."""
    try:
        if file and not file.closed:
            file.close()
    except Exception as e:
        print(f"Error closing log file: {e}", file=sys.__stderr__)  # Print to original stderr
