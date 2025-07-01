import spacy
import numpy as np
import pandas as pd
import random
from rdflib import Graph, Namespace
from rapidfuzz import fuzz, distance
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx 
# Define Namespaces
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
SKOSXL = Namespace("http://www.w3.org/2008/05/skos-xl#")
RDF = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")

print("Load RDF graph")
# Load RDF Graph
g = Graph()
g.parse("new_taxonomy_skos.xml", format="xml")

# Load SpaCy Model for embeddings
nlp = spacy.load("en_core_web_md")

# Function to compute weighted Jaccard similarity
def token_jaccard(a, b):
    set_a, set_b = set(a.split()), set(b.split())  # Tokenize and get sets
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0

# Function to compute cosine similarity for embeddings
def cosine_sim(vec1, vec2):
    if vec1 is None or vec2 is None:
        return 0
    vec1, vec2 = np.array(vec1).reshape(1, -1), np.array(vec2).reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]
def normalize_label(label):
    return label.lower().strip() if label else ""

hierarchy_graph = nx.DiGraph()

for concept in g.subjects(predicate=RDF.type, object=SKOS.Concept):
    for broader in g.objects(subject=concept, predicate=SKOS.broader):
        hierarchy_graph.add_edge(str(broader), str(concept))  # Directed edge from broader to narrower concept
concept_depths = {}
for concept in nx.topological_sort(hierarchy_graph):
    if concept not in concept_depths:
        concept_depths[concept] = 0  # Root concepts
    for child in hierarchy_graph.successors(concept):
        concept_depths[child] = concept_depths[concept] + 1

def hierarchical_distance(concept1, concept2):
    if concept1 == concept2:
        return 0  # Same concept
    try:
        boh = nx.shortest_path_length(hierarchy_graph, source=concept1, target=concept2)
        return boh
    except:
        return np.inf 
    
# Function to compute lowest common ancestor (LCA) depth
def lowest_common_ancestor_depth(concept1, concept2):
    try:
        ancestors1 = set(nx.ancestors(hierarchy_graph, concept1))
        ancestors2 = set(nx.ancestors(hierarchy_graph, concept2))
        common_ancestors = ancestors1 & ancestors2
        if not common_ancestors:
            return 0  # No common ancestor
        return max(concept_depths[ancestor] for ancestor in common_ancestors)
    except: 
        # print("Concepts "+ concept1 +concept2+ " don't have parents")
        return 0

# Function to compute number of shared ancestors
def shared_ancestor_count(concept1, concept2):
    try:
        return len(set(nx.ancestors(hierarchy_graph, concept1)) & set(nx.ancestors(hierarchy_graph, concept2)))
    except:
        # print("Concepts "+ concept1 +concept2+ " don't have parents")
        return 0

# Function to compute hierarchical overlap ratio
def hierarchical_overlap_ratio(concept1, concept2):
    try:
        shared_ancestors = shared_ancestor_count(concept1, concept2)
        max_depth = max(concept_depths.get(concept1, 0), concept_depths.get(concept2, 0))
        return shared_ancestors / max_depth if max_depth > 0 else 0
    except:
        # print("Concepts "+ concept1 +concept2+ " don't have parents")
        return 0

# Function to compute parent similarity
def parent_similarity(concept1, concept2, embeddings):
    parents1 = list(g.objects(subject=concept1, predicate=SKOS.broader))
    parents2 = list(g.objects(subject=concept2, predicate=SKOS.broader))
    
    if not parents1 or not parents2:
        return 0
    
    similarities = [cosine_sim(embeddings.get(str(p1), np.zeros(300)), embeddings.get(str(p2), np.zeros(300)))
                    for p1 in parents1 for p2 in parents2]
    
    return max(similarities) if similarities else 0

# Function to compute sibling score
def sibling_score(concept1, concept2):
    parents1 = set(g.objects(subject=concept1, predicate=SKOS.broader))
    parents2 = set(g.objects(subject=concept2, predicate=SKOS.broader))
    return 1 if parents1 & parents2 else 0

# Data structure to store concept features
concepts = []
concept_labels = {}
concept_embeddings = {}

print('Extract concepts')
# Extract concepts and compute features
for concept in g.subjects(predicate=RDF.type, object=SKOS.Concept):
    label_resource = g.value(subject=concept, predicate=SKOSXL.prefLabel)
    preferred_label = normalize_label(str(g.value(subject=label_resource, predicate=SKOSXL.literalForm))) if label_resource else None
    
    if not preferred_label:
        continue
    
    concepts.append(str(concept))
    concept_labels[str(concept)] = preferred_label
    concept_embeddings[str(concept)] = nlp(preferred_label).vector if preferred_label else np.zeros(300)

# Compute pairwise features
pairwise_data = []
#preselection of negatives, in such and such way and then avoid doing comparison between every concept. Try deviding on branches
for i, concept1 in enumerate(concepts):
    for j, concept2 in enumerate(concepts):
        if i >= j:
            continue  # Avoid redundant pairs and self-comparisons
        if j>= 10000:
            break

        label1, label2 = concept_labels[concept1], concept_labels[concept2]
        embed1, embed2 = concept_embeddings[concept1], concept_embeddings[concept2]
        #find heuristic for dividing into branches
        lexical_features = {
            "Concept1": concept1,
            "Concept2": concept2,   
            "Label1": label1,
            "Label2": label2,
            "Exact String Match": int(label1.lower() == label2.lower()),
            "Partial String Match": fuzz.partial_ratio(label1, label2) / 100,
            "Weighted Jaccard": token_jaccard(label1, label2),
            "Cosine Similarity": cosine_sim(embed1, embed2),
            "Levenshtein Distance": distance.Levenshtein.distance(label1, label2),
            "Depth Difference": abs(concept_depths.get(concept1, 0) - concept_depths.get(concept2, 0)),
            "Common Ancestor Depth": lowest_common_ancestor_depth(concept1, concept2),
            "Shared Ancestor Count": shared_ancestor_count(concept1, concept2),
            "Hierarchical Overlap Ratio": hierarchical_overlap_ratio(concept1, concept2),
            "Parent Similarity": parent_similarity(concept1, concept2, concept_embeddings),
            "Sibling Score": sibling_score(concept1, concept2)
        }
        
        lexical_features["Hierarchical Distance"] = hierarchical_distance(concept1, concept2)

        # Structural features
        broader1 = set(g.objects(subject=concept1, predicate=SKOS.broader))
        broader2 = set(g.objects(subject=concept2, predicate=SKOS.broader))
        narrower1 = set(g.objects(subject=concept1, predicate=SKOS.narrower))
        narrower2 = set(g.objects(subject=concept2, predicate=SKOS.narrower))
        
        lexical_features.update({
            "Shared Broader Concept": 1 if broader1 & broader2 else 0,
            "Shared Narrower Concept": 1 if narrower1 & narrower2 else 0,
        })
        
        # External knowledge features
        cross_refs1 = set(g.objects(subject=concept1, predicate=SKOS.exactMatch))
        # print('cross refs 1',concept1)
        cross_refs2 = set(g.objects(subject=concept2, predicate=SKOS.exactMatch))
        # print('cross refs 2',cross_refs2)
        lexical_features["Cross-Reference Match"] = 1 if cross_refs1 & cross_refs2 else 0
        if lexical_features["Cross-Reference Match"] == 1:
            print('hoorray')
        pairwise_data.append(lexical_features)
    if i>= 10000:
        break
# Convert to DataFrame
print('Save and show features')
df_features = pd.DataFrame(pairwise_data)
df_features.to_csv("pairwise_concept_features.csv", index=False)
# Display extracted features
# tools.display_dataframe_to_user(name="Pairwise Concept Features", dataframe=df_features)
print(df_features.head())
# Display table using Seaborn
plt.figure(figsize=(10, 6))
sns.heatmap(df_features.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()
