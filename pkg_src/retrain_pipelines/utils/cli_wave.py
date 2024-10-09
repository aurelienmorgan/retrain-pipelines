
import os
import sys
import time
import math

from colorama import Fore, Back, Style, init

from IPython import get_ipython
from IPython.display import display, clear_output, HTML

# Initialize colorama
init(strip=False)

def _create_wave_frame_html(text, t, width):
    # Generate a wave effect using HTML span elements for Jupyter
    wave_frame = []
    text_length = len(text)

    for x in range(width):
        char_index = x % text_length
        wave = math.sin(x / 5 + t) + 1

        if wave > 1.8:
            wave_frame.append(
                f"<span class='wave-magenta'>{text[char_index].upper()}</span>")
        else:
            wave_frame.append(
                f"<span class='wave-yellow'>{text[char_index]}</span>")

    return ''.join(wave_frame)

def _create_wave_frame_cli(text, t, width):
    # Generate a wave effect using colorama for CLI
    wave_frame = []
    text_length = len(text)

    for x in range(width):
        char_index = x % text_length
        wave = math.sin(x / 5 + t) + 1

        if wave > 1.8:
            wave_frame.append(
                Fore.MAGENTA + text[char_index].upper() + Style.RESET_ALL)
        else:
            wave_frame.append(
                Fore.YELLOW + text[char_index] + Style.RESET_ALL)

    return ''.join(wave_frame)

def animate_wave(text, duration=20, fps=15):
    """
    """

    if (
        os.getenv('launched_from_magic', None) or
        os.getenv('launched_from_cli', None)
    ):
        import io

        # Redirect stdout
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout

        print(Back.BLACK + Fore.YELLOW + f" {text} " + Style.RESET_ALL)

        sys.stdout = old_stdout
        output = new_stdout.getvalue()

        print(output, end="", flush=True)

        return

    in_jupyter = get_ipython() is not None
    if in_jupyter:
        # CSS styles for Jupyter output (unconditional black background)
        styles = """
        <style>
        .wave-container {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 5px;
            background-color: black;
            font-family: monospace;
        }
        .wave-yellow { color: #FFD700; }
        .wave-magenta { color: #FF00FF; font-weight: bold; }
        </style>
        """

    width = len(text)  # Width of the animation
    for t in range(int(duration * fps)):
        if in_jupyter:
            # Generate wave frame for Jupyter and update display
            frame = _create_wave_frame_html(text, t * 0.2, width)
            clear_output(wait=True)
            display(HTML(
                styles + f"<div class='wave-container'>{frame}</div>"))
        else:
            # Generate wave frame for CLI and print in place (unchanged)
            frame = _create_wave_frame_cli(text, t * 0.2, width)
            print(f"\r{frame}", end="", flush=True)

        time.sleep(1 / fps)

    # Final frame in lowercase yellow
    final_frame = text.lower()

    if in_jupyter:
        # Final display in Jupyter (full yellow on black background)
        clear_output(wait=True)
        display(HTML(styles + f"<div class='wave-container'><span class='wave-yellow'>{final_frame}</span></div>"))
    else:
        # Final print in CLI (yellow, unchanged)
        print(f"\r{Fore.YELLOW}{final_frame}{Style.RESET_ALL}",
              end="\n", flush=True)

