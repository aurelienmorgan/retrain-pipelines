
import textwrap, io

import networkx as nx
from metaflow import FlowSpec
from graphviz import Source, Digraph


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

