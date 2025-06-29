from rich.text import Text
from wcwidth import wcswidth


def framed_rich_log_str(
    rich_markup: str,
    border_color: str = "white",
    font_color: str = "default"
) -> str:
    """Puts a colored rounded border around message to be rich-logged.

    Params:
        - rich_markup (str):
            the formatted messaged to be logged
        - border_color (str):
            the color of the border line
        - font_color (str):
            the default font color
            (for substrings not marked for specific coloring)
    """
    markup_lines = rich_markup.splitlines()

    # Apply default font color to each line, if not already colored
    colored_lines = [
        f"[{font_color}]{line}[/]" for line in markup_lines
    ]

    text_block = Text.from_markup('\n'.join(colored_lines))
    plain_lines = text_block.plain.splitlines()
    max_width = max(wcswidth(line) for line in plain_lines)

    # Frame characters
    tl, tr = f"[{border_color}]╭[/]", f"[{border_color}]╮[/]"
    bl, br = f"[{border_color}]╰[/]", f"[{border_color}]╯[/]"
    hor = f"[{border_color}]─[/]"
    vert = f"[{border_color}]│[/]"

    top = tl + hor * (max_width + 2) + tr
    bottom = bl + hor * (max_width + 2) + br

    framed_lines = [top]

    for plain_line, markup_line in zip(plain_lines, colored_lines):
        padding = max_width - wcswidth(plain_line)
        framed_lines.append(f"{vert} {markup_line}{' ' * padding} {vert}")

    framed_lines.append(bottom)

    return '\n'.join(framed_lines)

