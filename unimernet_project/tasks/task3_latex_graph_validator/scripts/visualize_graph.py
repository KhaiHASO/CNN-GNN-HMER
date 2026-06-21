import os
import sys
import networkx as nx
import matplotlib.pyplot as plt

# Add current folder to path to import tokenizer & builder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from latex_tokenizer import tokenize
from latex_graph_builder import build_latex_graph

def get_tree_pos(G):
    """Calculate node positions for tree structure (root at top, depth decreases downwards)."""
    pos = {}
    levels = {}
    for node, data in G.nodes(data=True):
        depth = data.get('depth', 0)
        levels.setdefault(depth, []).append(node)
        
    for depth, nodes in sorted(levels.items()):
        num_nodes = len(nodes)
        for idx, node in enumerate(sorted(nodes)):
            # Distribute nodes horizontally at each depth level
            x = (idx + 1) / (num_nodes + 1)
            y = 1.0 - (depth * 0.2)
            pos[node] = (x, y)
    return pos

def visualize_and_save(latex_str, filename, title):
    """Build graph, draw it using matplotlib, and save as PNG."""
    print(f"Visualizing graph for: {latex_str} -> {filename}")
    
    # Check if there is an error in parsing (to show partial or token graph)
    try:
        G = build_latex_graph(latex_str)
    except Exception as e:
        # If parsing fails, build a simple sequential graph of tokens for visualization
        print(f"Parsing failed, building fallback token graph: {e}")
        G = nx.DiGraph()
        tokens = tokenize(latex_str)
        for idx, t in enumerate(tokens):
            G.add_node(idx, label=t['token'], type=t['type'], depth=0)
            if idx > 0:
                G.add_edge(idx-1, idx, relation='sequential')
                
    plt.figure(figsize=(10, 6))
    
    # Calculate positions
    pos = get_tree_pos(G)
    if not pos:
        pos = nx.spring_layout(G)
        
    # Get node labels and colors
    labels = {node: data.get('label', '') for node, data in G.nodes(data=True)}
    node_types = [data.get('type', 'token') for _, data in G.nodes(data=True)]
    
    # Color map for different node types
    colors = []
    for nt in node_types:
        if nt == 'frac':
            colors.append('#ff7675') # Coral
        elif nt == 'sqrt':
            colors.append('#74b9ff') # Blue
        elif nt in ('sup', 'sub'):
            colors.append('#a29bfe') # Purple
        elif nt in ('group', '{}', '()', '[]'):
            colors.append('#ffeaa7') # Yellow
        elif nt == 'expression':
            colors.append('#55efc4') # Mint green
        else:
            colors.append('#dfe6e9') # Light grey
            
    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=1200, edgecolors='black')
    
    # Draw edges with different styles
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        rel = data.get('relation', '')
        edge_labels[(u, v)] = rel
        
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=15, width=1.5, edge_color='#636e72')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, font_color='#2d3436')
    
    plt.title(f"{title}\n{latex_str}", fontsize=12, pad=15)
    plt.axis('off')
    plt.tight_layout()
    
    # Save figure
    os.makedirs("tasks/task3_latex_graph_validator/reports/figures", exist_ok=True)
    out_path = os.path.join("tasks/task3_latex_graph_validator/reports/figures", filename)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

def main():
    samples = [
        ("x + 1", "sample_01_simple.png", "1. Biểu thức đơn giản (Simple Expression)"),
        (r"\frac{x+1}{y}", "sample_02_fraction.png", "2. Phân số (Fraction)"),
        (r"\sqrt{x^2 + y^2}", "sample_03_sqrt.png", "3. Căn thức (Square Root)"),
        (r"x_j^2", "sample_04_supsub.png", "4. Chỉ số trên/dưới (Superscript/Subscript)"),
        (r"\frac{x+1", "sample_05_invalid.png", "5. Biểu thức lỗi (Invalid LaTeX)")
    ]
    
    for latex, fname, title in samples:
        visualize_and_save(latex, fname, title)
        
    print("\nGraph visualization complete! Check figures in tasks/task3_latex_graph_validator/reports/figures/")

if __name__ == "__main__":
    main()
