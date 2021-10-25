from collections import defaultdict
import re
import graphviz


STYLES = {
    'invar': {
        'shape': 'oval',
        'style': 'filled',
        'color': 'green',
    },
    'primitive': {
        'shape': 'rect',
        'style': 'filled',
        'color': 'lightblue',
    },
    'plus': {
        'shape': 'circle',
    },
    'outvar': {
        'shape': 'oval',
        'style': 'filled',
        'color': 'red',
    },
    'constant': {
        'shape': 'oval',
        'style': 'filled',
        'color': 'yellow',
    }
}

EDGE_STYLES = {
    'param': {
        'weight': ' 1.0',
        'style': 'dashed',
        'group': 'B'
    },
    'constant': {
        'weight': ' 10.0',
        'style': 'dashed',
        'group': 'A'
    },
    'primitive': {
        'weight': '10.0',
        'group': 'A'
    },
    'param_layout': {
        'weight': '100.0',
        'style': 'dotted',
        'constraint': 'false',
        'group': 'C'
    },
}

is_var = re.compile("[a-z]+")


def make_graph(jxpr, name='', render_add_as_plus=True) -> graphviz.Digraph:
    eqns = jxpr.jaxpr.eqns
    invars = jxpr.jaxpr.invars
    outvars = jxpr.jaxpr.outvars
    dot = graphviz.Digraph(name=name,
                           engine='dot',
                           graph_attr={'overlap_shrink': 'true'})

    def primitive_node_name(eq, i):
        return f"{str(eq.primitive)}{i}{name}"

    # Track and make nodes for input and output variables.
    real_vars = set([str(v) for v in invars + outvars])
    dot.node(f"{invars[-1]}{name}", str(invars[-1]), STYLES['invar'])
    for invar in invars[:-1]:
        dot.node(f"{invar}{name}", str(invar), STYLES['invar'])
    for outvar in outvars:
        dot.node(f"{outvar}{name}", str(outvar), STYLES['outvar'])

    # Collapse the internal "temp" variables to become labels on op->op edges.
    edge_vars = defaultdict(lambda: [])
    for i, eq in enumerate(jxpr.jaxpr.eqns):
        node_name = primitive_node_name(eq, i)
        for outvar in eq.outvars:
            edge_vars[str(outvar)] = []
        for invar in eq.invars:
            edge_vars[str(invar)].append(node_name)

    def primitive_node(sub_graph, eq, i):
        node_name = primitive_node_name(eq, i)
        if render_add_as_plus:
            if str(eq.primitive) == 'add':
                sub_graph.node(node_name, '+', STYLES['plus'])
                return
        sub_graph.node(node_name, str(eq.primitive), STYLES['primitive'])

    with dot.subgraph(name=f'cluster_{name}primitives',
                      graph_attr=dict(style='dashed',
                                      color='grey')) as sub_graph:
        for i, eq in enumerate(eqns):
            primitive_node(sub_graph, eq, i)
            for invar in eq.invars:
                if not is_var.match(str(invar)):
                    sub_graph.node(f"const_{name}{i}",
                                   f"{float(str(invar)):0.2}",
                                   STYLES['constant'])

    # Trace all of the equations (i.e., NN ops) to connect the graph
    for i, eq in enumerate(jxpr.jaxpr.eqns):
        node_name = primitive_node_name(eq, i)
        #dot.node(node_name, str(eq.primitive), STYLES['primitive'])
        for invar in eq.invars:
            if not is_var.match(str(invar)):
                dot.node(f"const_{name}{i}", f"{float(str(invar)):0.2}",
                         STYLES['constant'])
                dot.edge(f"const_{name}{i}", node_name, '',
                         EDGE_STYLES['constant'])
            if str(invar) in real_vars:
                dot.edge(f"{invar}{name}", node_name, '', EDGE_STYLES['param'])
        for outvar in eq.outvars:
            if str(outvar) in real_vars:
                dot.edge(node_name, f"{outvar}{name}", '', EDGE_STYLES['param'])
            else:
                for out_primitive in edge_vars[str(outvar)]:
                    dot.edge(node_name, out_primitive, '',
                             EDGE_STYLES['primitive'])
    return dot
