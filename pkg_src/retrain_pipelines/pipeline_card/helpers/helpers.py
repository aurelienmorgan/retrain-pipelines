
import re
import base64

import pandas as pd

from io import BytesIO
from matplotlib.figure import Figure

from metaflow import FlowSpec


def apply_args_color_format(
  python_command_str: str
) -> str:
    """
    Apply html python-style coloring
    to python command strings.

    Params:
        - python_command_str (str):
            the raw python command string

    Results:
        - (str)
    """

    # Define the regex pattern to match substrings
    pattern = r'([^()]*\()([^()]+)(\).*)'
    match = re.match(pattern, python_command_str)

    formatted_args = []
    before_parentheses = match.group(1)
    parameters_str = match.group(2)
    after_parentheses = match.group(3)

    # print("Before Parentheses:", before_parentheses)
    # print("Between Parentheses:", parameters_str)
    # print("After Parentheses:", after_parentheses)

    for parameter_str in parameters_str.split(','):
        # print(_apply_arg_color_format(parameter_str))
        formatted_args.append(_apply_arg_color_format(parameter_str))

    result = (
        before_parentheses +
        ', '.join(formatted_args) +
        after_parentheses
    )

    return result


def _apply_arg_color_format(
    argument_str:str
) -> str:
    """
    encapsulates the input arg between html font tags
    for font coloring.

    Params:
        - argument_str (str)
          a string representation of an argument
          ex.: `var_name=0`       # for named int arguments
          ex.: "path/to/a/dir"   # for unnamed string argument

    Usage:
        input_strings = [
            '"path/to/a/dir"',
            'path/to/a/dir',
            'var_name="0"',
            'var_name=0',
            '0'
        ]
        for input_string in input_strings:
            print(_apply_arg_color_format(input_string))
    """

    pattern = r'((.+\s*=)|([^,]+))(\s*.+)?'
    font_tag_head = \
        "<font color=\"#eb5656;\">"

    def match_replace(match):
        if match.group(3):
            # case 'not a named parameter
            return font_tag_head + match.group(3) + "</font>"
        else:
            return match.group(1) + font_tag_head + match.group(4) + "</font>"

    #uncomment below to debug regex groups
    # re.sub(pattern, lambda match:
    #            print(
    #                f'Group 1: {match.group(1)}\n' +
    #                f'Group 2: {match.group(2)}\n' +
    #                f'Group 3: {match.group(3)}\n' +
    #                f'Group 4: {match.group(4)}'
    #            ),
    #        input_string)

    result = re.sub(
        pattern,
        match_replace,
        argument_str
    )

    return result


def highlight_min_max_cells(
    df: pd.DataFrame
) -> str:
    """
    Convert dataframe into stylized html.
    Add green/red coloring
    for min/max numeric value per column.
    Also formats floats to
    up to 3 decimal digits max.
    Also, the table css class is assigned
    value `class="wide"`.

    Params:
        - df (pd.DataFrame)

    Results:
        - (str)
            html table
    """
    df = df.copy()

    def _format_float(x):
        if isinstance(x, (float, int)):
            return '{:.3f}'.format(x).rstrip('0').rstrip('.')
        return x

    df = df.map(_format_float)

    def _highlight_min_max(df):
        """add green/red coloring
        for min/max numeric value per column"""
        styles = pd.DataFrame('', index=df.index,
                              columns=df.columns)
        for col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            styles.loc[df[col] == min_val, col] = \
                'background-color: rgba(255, 0, 0, 0.2)'
            styles.loc[df[col] == max_val, col] = \
                'background-color: rgba(0, 255, 0, 0.2)'
        return styles

    styled_df = df.style.apply(_highlight_min_max,
                               axis=None)
    styled_table = \
        styled_df.to_html(table_attributes='class="wide"',
                          escape=False, index = False)

    return styled_table


def fig_to_base64(
    plt_fig: Figure,
    extra_tight: bool = False
) -> str:
    """
    Converts a figure into base64-encoded png
    image data.
    Can serve for image embedding
    into a portable html file.

    Params:
        - plt_fig (Figure)
            the figure to encode
        - extra_tight (bool)
            go against the natural tendance
            to add margin whiule saving
            with a bytes_io object.

    Results:
        - str
    """

    bytes_io_obj = BytesIO()
    if not extra_tight:
        plt_fig.savefig(bytes_io_obj, format='png')
    else:
        plt_fig.savefig(bytes_io_obj, format='png',
                        bbox_inches='tight',
                        pad_inches=0.05)
    bytes_io_obj.seek(0)
    base64_png = base64.b64encode(
                    bytes_io_obj.read()).decode()

    return base64_png

