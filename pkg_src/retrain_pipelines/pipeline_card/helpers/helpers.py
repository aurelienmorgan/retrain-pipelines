
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


def mf_dag_svg(flow_spec: FlowSpec) -> str:
    """
    Generate a Metaflow flow DAG representation
    in svg format, somewhat stylized.

    Params:
        - flow_spec (FlowSpec):
            the source for the DAG to be generated.

    Results:
        - (str)
    """
    import textwrap, io

    import networkx as nx
    from graphviz import Source, Digraph

    #########################################
    #       scan the flow_spec graph        #
    #########################################
    flow_graph_dict =  flow_spec._graph.output_steps()[0]

    # TODO  -  update to handle cases with 'foreach' loops
    #          that are nested and/or per (flow) subbranch
    edges = []
    nodes = []
    inner_foreach = 0
    for key in flow_graph_dict:
        step = flow_graph_dict[key]
        if 'join' == step['type']:
            inner_foreach -=1
        nodes.append(
            (
                key,
                (inner_foreach > 0), textwrap.fill(
                                        step['doc']
                                        .replace('\n', ' '),
                                        50)
            )
        )
        if 'split-foreach' == step['type']:
            inner_foreach +=1
        if step['next']:
            for next_step in step['next']:
                edges.append((key, next_step))
                if (inner_foreach > 0):
                    edges.append((key, next_step))
                    edges.append((key, next_step))
    #########################################

    
    #########################################
    # build DAG graph from flow-spec object #
    #########################################
    G = nx.MultiDiGraph(type="digraph", name='FlowDag')
    box3d_nodes = []
    for i, node in enumerate(nodes):
        G.add_node(node[0], label=(
            "<" +
            "<table border=\"0\" cellborder=\"0\" cellspacing=\"0\" cellpadding=\"0\">" +
            f"<tr><td align=\"center\"><b>{node[0]} </b></td></tr>" +
            "</table>" +
            ">"
        ))
        xlabel = node[2].replace('\n', '<br />')
        if xlabel:
            nx.set_node_attributes(G, {node[0]: '10em'}, name="fontsize")
            nx.set_node_attributes(G,
                                   {node[0]: (
               "<" +
               "<table border=\"1\" cellborder=\"0\" cellspacing=\"0\" cellpadding=\"5\">" +
               f"<tr><td><font color=\"#AAFF00\">{xlabel}</font></td></tr>" +
               "</table>" +
               ">"
                                   )},
                                   name="xlabel")
        if node[1]:
            nx.set_node_attributes(G, {node[0]: 'box3d'},
                                   name="shape")
            nx.set_node_attributes(G, {node[0]: 'filled'},
                                   name="style")
            nx.set_node_attributes(G, {node[0]: 'orange'},
                                   name="fillcolor")
            nx.set_node_attributes(G, {node[0]: '#e67e00'},
                                   name="fontcolor")
            nx.set_node_attributes(G,
                                   {node[0]: (
               "<" +
               "<table border=\"0\" cellborder=\"0\" cellspacing=\"0\" cellpadding=\"0\">" +
               f"<TR><TD>\u00A0 <U><B>{node[0]}</B></U> \u00A0 \u00A0 </TD></TR>" +
               "</table>" +
               ">"
                                   )},
                                   name="label")
            box3d_nodes.append(node[0])
        else:
            nx.set_node_attributes(G, {node[0]: '\"gold:brown\"'},
                                   name="fillcolor")
            nx.set_node_attributes(G, {node[0]: 'radial'},
                                   name="style")
            nx.set_node_attributes(G, {node[0]: '180'},
                                   name="gradientangle")

    for edge in edges:
        G.add_edge(edge[0], edge[1], color="gold")
    # import matplotlib.pyplot as plt
    # nx.draw(G, with_labels = True)
    # plt.savefig("filename.png")
    # plt.show()
    #########################################


    #########################################
    #         export to DOT string          #
    #########################################
    # Create a StringIO buffer
    buffer = io.StringIO()
    nx.drawing.nx_pydot.write_dot(G, buffer)
    buffer_contents = buffer.getvalue()
    buffer.close()
    # print(buffer_contents)
    #########################################


    #########################################
    #           refine formatting           #
    #     (tranparent background, etc.)     #
    #########################################
    s = Source(buffer_contents, format='svg')

    g = Digraph()
    source_lines = str(s).splitlines()
    # Remove 'digraph graphname {'
    source_lines.pop(0)
    # Remove the closing brackets '}'
    source_lines.pop(-1)
    # Append the nodes to body
    g.body += source_lines

    #g.graph_attr['rankdir'] = 'LR'
    g.graph_attr['bgcolor'] = "#00ffff20" # alpha transparency
    g.node_attr['labelloc'] = 'c'
    g.node_attr['margin'] = '0'
    g.node_attr['fontname'] = 'Arial'
    g.node_attr['fontsize'] = '11em'
    #########################################

    graph_svg = g.pipe(format='svg').decode('utf-8')

    #########################################
    #     highlight text in box3d_nodes     #
    #########################################
    # inspired from 
    from lxml.etree import fromstring, tostring, Element, SubElement
    root = fromstring(graph_svg.encode())
    defs = root.find('defs')
    if defs is None:
        defs = Element('defs')
        root.insert(0, defs)  # Add <defs> as the first child of <svg>
    filter_elem = SubElement(defs, 'filter', x="0", y="0",
                             width="1", height="1", id="solid")

    SubElement(filter_elem, 'feFlood', **{'flood-color': 'yellow',
                                          'result': 'bg'})
    feMerge_elem = SubElement(filter_elem, 'feMerge')
    SubElement(feMerge_elem, 'feMergeNode', **{'in': 'bg'})
    SubElement(feMerge_elem, 'feMergeNode', **{'in': 'SourceGraphic'})

    for text_elem in root.findall('.//{http://www.w3.org/2000/svg}text'):
        if text_elem.text in box3d_nodes:
            text_elem.set('filter', 'url(#solid)')

    graph_svg = tostring(root, encoding='utf-8', method='xml'
                        ).decode('utf-8')
    #########################################

    return graph_svg


