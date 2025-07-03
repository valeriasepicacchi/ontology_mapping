import random
import torch
import spacy
import numpy as np
import networkx as nx
import pandas as pd
from rdflib import Graph, Namespace
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

''' This code was used for the implementation of a GNN on a company owned dataset and is therefore not reproducible '''

# Define Namespaces
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
SKOSXL = Namespace("http://www.w3.org/2008/05/skos-xl#")
RDF = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")

g = Graph()
g.parse('dataset', format="xml")  

# Initialize NetworkX Graph
G = nx.DiGraph()

# Extract Concepts and Relationships
for concept in g.subjects(predicate=RDF.type, object=SKOS.Concept):
    label_resource = g.value(subject=concept, predicate=SKOSXL.prefLabel)
    preferred_label = g.value(subject=label_resource, predicate=SKOSXL.literalForm)
    G.add_node(str(concept), label=str(preferred_label) if preferred_label else None)

    for broader in g.objects(subject=concept, predicate=SKOS.broader):
        G.add_edge(str(broader), str(concept), relation="broader")

    for narrower in g.objects(subject=concept, predicate=SKOS.narrower):
        G.add_edge(str(concept), str(narrower), relation="narrower")

    for related in g.objects(subject=concept, predicate=SKOS.related):
        G.add_edge(str(concept), str(related), relation="related")

print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
nlp = spacy.load("en_core_web_md")

# Generate Node Embeddings
node_embeddings = {}
for node in G.nodes():
    label = G.nodes[node].get("label", "")
    node_embeddings[node] = nlp(label).vector if label else np.zeros((300,))

# Ensure all nodes have the same attributes
required_attributes = {"label": None, "embedding": np.zeros(300)}
for node in G.nodes:
    for attr, default_value in required_attributes.items():
        if attr not in G.nodes[node]:
            G.nodes[node][attr] = default_value

df_embeddings = pd.DataFrame.from_dict(node_embeddings, orient="index")

# Convert NetworkX graph to PyTorch Geometric format
graph_data = from_networkx(G)
graph_data.x = torch.tensor(df_embeddings.values, dtype=torch.float)

# Create Positive & Negative Edges
edges = list(G.edges())
edge_index = torch.tensor([[list(G.nodes()).index(e[0]), list(G.nodes()).index(e[1])] for e in edges], dtype=torch.long).t()

# Generate Negative Edges (randomly sampled)
all_nodes = list(G.nodes())
negative_edges = set()
while len(negative_edges) < len(edges):
    a, b = random.sample(all_nodes, 2)
    if (a, b) not in G.edges() and (b, a) not in G.edges():
        negative_edges.add((a, b))

neg_edge_index = torch.tensor([[list(G.nodes()).index(e[0]), list(G.nodes()).index(e[1])] for e in negative_edges], dtype=torch.long).t()

# Train-Test Split
pos_train, pos_test = train_test_split(edge_index.t().tolist(), test_size=0.2, random_state=42)
neg_train, neg_test = train_test_split(neg_edge_index.t().tolist(), test_size=0.2, random_state=42)

pos_train, pos_test = torch.tensor(pos_train).t(), torch.tensor(pos_test).t()
neg_train, neg_test = torch.tensor(neg_train).t(), torch.tensor(neg_test).t()

train_edges = torch.cat([pos_train, neg_train], dim=1)
train_labels = torch.cat([torch.ones(pos_train.shape[1]), torch.zeros(neg_train.shape[1])])

test_edges = torch.cat([pos_test, neg_test], dim=1)
test_labels = torch.cat([torch.ones(pos_test.shape[1]), torch.zeros(neg_test.shape[1])])

print(f"Train edges: {train_edges.shape}, Test edges: {test_edges.shape}")

# Define Graph Neural Network (GNN) Model
class GCNLinkPredictor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GCNLinkPredictor, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, data, edge_index):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = self.conv2(x, data.edge_index)
        edge_embeddings = x[edge_index[0]] * x[edge_index[1]]
        return torch.sigmoid(self.fc(edge_embeddings)).squeeze()

# Initialize Model
model = GCNLinkPredictor(input_dim=300, hidden_dim=128)

# Train Model
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.BCELoss()

epochs = 700
model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    preds = model(graph_data, train_edges)
    loss = loss_fn(preds, train_labels)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Evaluate Model
model.eval()
with torch.no_grad():
    test_preds = model(graph_data, test_edges)
auc = roc_auc_score(test_labels.numpy(), test_preds.numpy())
print(f"AUC Score: {auc:.4f}")

# Predict Relationship Between Two Concepts
def predict_relation(concept1, concept2):
    with torch.no_grad():
        edge = torch.tensor([[list(G.nodes()).index(concept1)], [list(G.nodes()).index(concept2)]])
        score = model(graph_data, edge).item()
        return f"Relationship Score: {score:.4f} (Higher = More Likely Related)"

# Example Prediction
example_concept1 = list(G.nodes())[0]  
example_concept2 = list(G.nodes())[1]
print(predict_relation(example_concept1, example_concept2))


