from typing import List, Dict, Tuple
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass
import torch
import torch.nn.functional as F

class ConceptVisualizer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def create_concept_graph(self, 
                           concepts: set,
                           relations: List[Tuple[str, str, str]],
                           min_connections: int = 1) -> nx.Graph:
        """Create NetworkX graph from concepts and relations"""
        G = nx.Graph()
        
        # Add nodes (concepts)
        for concept in concepts:
            G.add_node(concept)
        
        # Add edges (relations)
        for subj, rel, obj in relations:
            if subj in concepts and obj in concepts:
                G.add_edge(subj, obj, relationship=rel)
        
        # Remove nodes with too few connections
        nodes_to_remove = [node for node, degree in dict(G.degree()).items() 
                         if degree < min_connections]
        G.remove_nodes_from(nodes_to_remove)
        
        return G
    
    def visualize_concepts(self, 
                         concepts: set,
                         relations: List[Tuple[str, str, str]],
                         title: str = "Concept Map",
                         filename: str = "concept_map.svg"):
        """Create and save concept visualization"""
        plt.figure(figsize=(15, 10))
        
        # Create graph
        G = self.create_concept_graph(concepts, relations)
        
        # Set up layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                             node_color='lightblue',
                             node_size=2000,
                             alpha=0.7)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, 
                             edge_color='gray',
                             width=1,
                             alpha=0.5)
        
        # Add labels with word wrapping
        labels = {}
        for node in G.nodes():
            labels[node] = '\n'.join(self._wrap_text(node))
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        # Add edge labels (relationships)
        edge_labels = nx.get_edge_attributes(G, 'relationship')
        wrapped_edge_labels = {k: '\n'.join(self._wrap_text(v)) 
                             for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(G, pos, 
                                   edge_labels=wrapped_edge_labels,
                                   font_size=6)
        
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        
        # Save visualization
        plt.savefig(filename, format='svg', bbox_inches='tight')
        plt.close()
        
    def _wrap_text(self, text: str, max_line_length: int = 20) -> List[str]:
        """Wrap text to fit in nodes/edges"""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= max_line_length:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
                
        if current_line:
            lines.append(' '.join(current_line))
            
        return lines