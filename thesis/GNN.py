from rdflib import Graph, Namespace
import networkx as nx
import pandas as pd

# Define SKOS Namespaces
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
SKOSXL = Namespace("http://www.w3.org/2008/05/skos-xl#")
RDF = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")

# Load RDF Graph
g = Graph()
g.parse("emtree_release_202501.xml", format="xml")  # Update with your actual file

# Initialize NetworkX graph
G = nx.DiGraph()

# Extract Concepts and Relationships
for concept in g.subjects(predicate=RDF.type, object=SKOS.Concept):
    # Preferred Label
    label_resource = g.value(subject=concept, predicate=SKOSXL.prefLabel)
    preferred_label = g.value(subject=label_resource, predicate=SKOSXL.literalForm)

    # Add Node
    G.add_node(str(concept), label=str(preferred_label) if preferred_label else None)

    # Add Hierarchical Relationships
    for broader in g.objects(subject=concept, predicate=SKOS.broader):
        G.add_edge(str(broader), str(concept), relation="broader")

    for narrower in g.objects(subject=concept, predicate=SKOS.narrower):
        G.add_edge(str(concept), str(narrower), relation="narrower")

    for related in g.objects(subject=concept, predicate=SKOS.related):
        G.add_edge(str(concept), str(related), relation="exactMatch")

# Display Graph Statistics
print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
import spacy
import numpy as np

# Load SpaCy Model
nlp = spacy.load("en_core_web_md")

# Generate Node Embeddings
node_embeddings = {}
for node in G.nodes():
    label = G.nodes[node].get("label", "")
    if label:
        node_embeddings[node] = nlp(label).vector
    else:
        node_embeddings[node] = np.zeros((300,))  # Fallback for missing labels

# Convert to DataFrame
df_embeddings = pd.DataFrame.from_dict(node_embeddings, orient="index")
print(df_embeddings.head())
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

# Convert NetworkX graph to PyTorch Geometric format
graph_data = from_networkx(G)

# Add Node Features (Embeddings)
graph_data.x = torch.tensor(df_embeddings.values, dtype=torch.float)

# Print Graph Data
print(graph_data)
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# Initialize Model
model = GCN(input_dim=300, hidden_dim=128, output_dim=6)  # Adjust based on clusters
print(model)
import torch.optim as optim

# Define Loss & Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

# Dummy labels (for supervised training)
labels = torch.randint(0, 6, (graph_data.num_nodes,))  # Random cluster assignments

# Training Loop
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(graph_data)  # Forward pass
    loss = loss_fn(out, labels)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Extract Node Embeddings
model.eval()
with torch.no_grad():
    embeddings = model(graph_data).numpy()

# Reduce to 2D
X_tsne = TSNE(n_components=2).fit_transform(embeddings)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap="viridis", marker="o")
plt.colorbar()
plt.title("Graph Neural Network Clustering of Taxonomy Concepts")
plt.show()
