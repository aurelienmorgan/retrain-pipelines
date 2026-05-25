import os
import time

from IPython.display import HTML, display, update_display
from rich.live import Live
from rich.text import Text

GOLD_HEX = "#c39c1a"
MAJENTA_HEX = "#9a2bab"


def _wave_cli(text: str, wave_length: int = 6, delay: float = 0.1, loops: int = 2):
    length = len(text)
    base_color = f"{GOLD_HEX} on black"
    wave_color = f"{MAJENTA_HEX} on black"

    with Live(refresh_per_second=50) as live:
        for loop in range(loops):
            if loop < loops - 1:
                # For all but last loop:
                # circular wrapping for smooth continuous wave (right to left)
                for i in range(length - 1, -1, -1):
                    rendered = Text()
                    for idx, char in enumerate(text):
                        dist = (i - idx) % length
                        if dist < wave_length:
                            if 2 <= dist <= length - 2:
                                display_char = char.upper()
                            else:
                                display_char = char
                            rendered.append(display_char, style=wave_color)
                        else:
                            rendered.append(char, style=base_color)
                    live.update(rendered)
                    time.sleep(delay)
            else:
                # For last loop:
                # linear wave sliding off the beginning, no wrapping
                for i in range(length + wave_length - 1, -1, -1):
                    rendered = Text()
                    for idx, char in enumerate(text):
                        if i - wave_length < idx <= i:
                            wave_pos = i - idx
                            if 2 <= wave_pos <= 4:
                                display_char = char.upper()
                            else:
                                display_char = char
                            rendered.append(display_char, style=wave_color)
                        else:
                            rendered.append(char, style=base_color)
                    live.update(rendered)
                    time.sleep(delay)

        # Final frame: all text in base color
        final_rendered = Text()
        for char in text:
            final_rendered.append(char, style=base_color)
        live.update(final_rendered)
        time.sleep(0.4)
        live.stop()


def _wave_notebook(text: str, wave_length: int = 6, delay: float = 0.1, loops: int = 2):
    # CSS styles for Jupyter output
    styles = f"""
    <style>
    .wave-container {{
      display: inline-block;
      padding: 5px 10px;
      border-radius: 5px;
      background-color: black;
      font-family: monospace;
    }}
    .wave-yellow {{ color: {GOLD_HEX}; }}
    .wave-magenta {{ color: {MAJENTA_HEX}; font-weight: bold; }}
    </style>
    """

    display_handle = display(HTML(styles + "<div class='wave-container'></div>"), display_id=True)

    frames = []
    length = len(text)
    for loop in range(loops):
        if loop < loops - 1:
            for i in range(length - 1, -1, -1):
                frame = ""
                for idx, char in enumerate(text):
                    dist = (i - idx) % length
                    if dist < wave_length:
                        if 2 <= dist <= length - 2:
                            display_char = char.upper()
                        else:
                            display_char = char
                        frame += f"<span class='wave-magenta'>{display_char}</span>"
                    else:
                        frame += f"<span class='wave-yellow'>{char}</span>"
                frames.append(frame)
        else:
            for i in range(length + wave_length - 1, -1, -1):
                frame = ""
                for idx, char in enumerate(text):
                    if i - wave_length < idx <= i:
                        wave_pos = i - idx
                        if 2 <= wave_pos <= 4:
                            display_char = char.upper()
                        else:
                            display_char = char
                        frame += f"<span class='wave-magenta'>{display_char}</span>"
                    else:
                        frame += f"<span class='wave-yellow'>{char}</span>"
                frames.append(frame)

    for frame in frames:
        update_display(
            HTML(styles + f"<div class='wave-container'>{frame}</div>"),
            display_id=display_handle.display_id,
        )
        time.sleep(delay)

    # Final frame
    final_frame = text.lower()
    update_display(
        HTML(
            styles
            + f"<div class='wave-container'><span class='wave-yellow'>{final_frame}</span></div>"
        ),
        display_id=display_handle.display_id,
    )


def animate_wave(text: str, wave_length: int = 6, delay: float = 0.1, loops: int = 2):
    """Animate wave effect for both CLI (rich) and Jupyter (HTML)."""
    from . import in_notebook

    if not (os.getenv("launched_from_magic", None) or os.getenv("launched_from_cli", None)):
        if in_notebook():
            _wave_notebook(text, wave_length, delay, loops)
        else:
            _wave_cli(text, wave_length, delay, loops)


# Example usage:
if __name__ == "__main__":
    animate_wave("Rich & Jupyter Wave!")
