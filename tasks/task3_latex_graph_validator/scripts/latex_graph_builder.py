import os
import sys
import networkx as nx

# Add current folder to path to import tokenizer & validator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from latex_tokenizer import tokenize
from latex_validator import LatexParser

class LatexGraphBuilder:
    """Class to convert LaTeX AST into a directed networkx graph."""
    def __init__(self):
        self.G = nx.DiGraph()
        self.node_id_counter = 0

    def get_new_id(self):
        self.node_id_counter += 1
        return self.node_id_counter

    def build(self, ast_root):
        self.G.clear()
        self.node_id_counter = 0
        if not ast_root:
            return self.G
        self._traverse(ast_root, parent_id=None, depth=0)
        return self.G

    def _traverse(self, ast_node, parent_id=None, depth=0):
        node_id = self.get_new_id()
        
        # Default node attributes
        label = ast_node.type
        pos = -1
        tok_type = ast_node.type
        
        if ast_node.type == 'token':
            label = ast_node.value['token']
            pos = ast_node.value['position']
            tok_type = ast_node.value['type']
        elif ast_node.type == 'group' and ast_node.value:
            label = ast_node.value
            
        self.G.add_node(node_id, label=label, type=tok_type, position=pos, depth=depth)
        
        # If there is a parent, determine relationship and add edge
        if parent_id is not None:
            parent_node = self.G.nodes[parent_id]
            parent_type = parent_node['type']
            
            edge_relation = 'parent_child'
            if parent_type == 'frac':
                # Numerator is the first outbound edge, Denominator is the second
                out_edges = [self.G.edges[e] for e in self.G.out_edges(parent_id)]
                frac_relations = [edg.get('relation') for edg in out_edges]
                if 'frac_numerator' not in frac_relations:
                    edge_relation = 'frac_numerator'
                else:
                    edge_relation = 'frac_denominator'
            elif parent_type == 'sqrt':
                edge_relation = 'sqrt_body'
            elif parent_type == 'sup':
                edge_relation = 'superscript'
            elif parent_type == 'sub':
                edge_relation = 'subscript'
            elif parent_type in ('group', '{}', '()', '[]'):
                edge_relation = 'inside_group'
                
            self.G.add_edge(parent_id, node_id, relation=edge_relation)
            
        # Traverse children
        child_ids = []
        for child in ast_node.children:
            child_id = self._traverse(child, parent_id=node_id, depth=depth + 1)
            child_ids.append(child_id)
            
        # Draw sequential edges between siblings at the same level
        if ast_node.type == 'expression' and len(child_ids) > 1:
            for i in range(len(child_ids) - 1):
                self.G.add_edge(child_ids[i], child_ids[i+1], relation='sequential')
                
        return node_id

def build_latex_graph(latex_str):
    """Tokenize, parse, and build networkx DiGraph for a LaTeX string."""
    tokens = tokenize(latex_str)
    parser = LatexParser(tokens)
    ast_root = parser.parse()
    builder = LatexGraphBuilder()
    graph = builder.build(ast_root)
    return graph

def compute_graph_stats(graph):
    """Compute statistics for the generated NetworkX graph."""
    if not graph or len(graph.nodes) == 0:
        return {
            "node_count": 0,
            "edge_count": 0,
            "max_depth": 0,
            "has_frac": 0,
            "has_sqrt": 0,
            "has_sup": 0,
            "has_sub": 0
        }
        
    node_count = len(graph.nodes)
    edge_count = len(graph.edges)
    
    depths = [data.get('depth', 0) for _, data in graph.nodes(data=True)]
    max_depth = max(depths) if depths else 0
    
    node_types = [data.get('type') for _, data in graph.nodes(data=True)]
    
    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "max_depth": max_depth,
        "has_frac": 1 if 'frac' in node_types else 0,
        "has_sqrt": 1 if 'sqrt' in node_types else 0,
        "has_sup": 1 if 'sup' in node_types else 0,
        "has_sub": 1 if 'sub' in node_types else 0
    }

if __name__ == "__main__":
    test_str = r"\frac{x^{2} + 15}{4} = \sqrt{y}"
    print(f"Building graph for: {test_str}")
    G = build_latex_graph(test_str)
    print(f"Nodes: {list(G.nodes(data=True))[:5]}")
    print(f"Edges: {list(G.edges(data=True))[:5]}")
    stats = compute_graph_stats(G)
    print(f"Stats: {stats}")
