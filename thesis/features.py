import spacy
import numpy as np
import pandas as pd
import networkx as nx
from rdflib import Graph, Namespace
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity

# Define Namespaces
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
SKOSXL = Namespace("http://www.w3.org/2008/05/skos-xl#")
RDF = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")

# Load RDF Graph
g = Graph()
g.parse("emtree_release_202501.xml", format="xml")  # Update with actual file

# Load SpaCy Model for embeddings
nlp = spacy.load("en_core_web_md")

# Function to compute weighted Jaccard similarity (WordMatcher)
def weighted_jaccard(a, b):
    set_a, set_b = set(str(a).split()), set(str(b).split())
    intersection = sum(len(word) for word in set_a & set_b)
    union = sum(len(word) for word in set_a | set_b)
    return intersection / union if union > 0 else 0

# Function to compute cosine similarity for embeddings (LLM Matcher)
def cosine_sim(vec1, vec2):
    if vec1 is None or vec2 is None:
        return 0
    vec1, vec2 = np.array(vec1).reshape(1, -1), np.array(vec2).reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]

# Data structure to store concept features
concept_data = []

# Extract concepts
for concept in g.subjects(predicate=RDF.type, object=SKOS.Concept):
    label_resource = g.value(subject=concept, predicate=SKOSXL.prefLabel)
    preferred_label = str(g.value(subject=label_resource, predicate=SKOSXL.literalForm)) if label_resource else None

    if not preferred_label:
        continue  # Skip if no label is available

    # Collect alternate labels (synonyms)
    synonyms = [str(syn) for syn in g.objects(subject=concept, predicate=SKOSXL.altLabel)]
    # print(len(synonyms))
    # Extract cross-references (if available)
    cross_refs = [str(ref) for ref in g.objects(subject=concept, predicate=SKOS.exactMatch)]
    
    # Compute lexical similarity scores
    label_embedding = nlp(preferred_label).vector if preferred_label else np.zeros(300)
    
    lexical_features = {
        "Concept": str(concept),
        "Preferred Label": preferred_label,
        "LexicalMatcher": 1 if any(preferred_label.lower() == syn.lower() for syn in synonyms) else 0,
        "SpacelessLexicalMatcher": 1 if any(preferred_label.replace(" ", "").lower() == syn.replace(" ", "").lower() for syn in synonyms) else 0,
        "WordMatcher": max(weighted_jaccard(preferred_label, syn) for syn in synonyms) if synonyms else 0,
        "StringMatcher": max(fuzz.partial_ratio(preferred_label, syn) / 100 for syn in synonyms) if synonyms else 0,
        "DirectXRefMatcher": 1 if cross_refs else 0,
        "MediatingMatcher": 0,  
        "MediatingXRefMatcher": 0,  
        "BackgroundKnowledgeMatcher": 0,  
        "ThesaurusMatcher": 1 if synonyms else 0,
        "LLM Matcher": max(cosine_sim(label_embedding, nlp(str(syn)).vector) for syn in synonyms) if synonyms else 0
    }

    # Append data to list
    concept_data.append(lexical_features)

# Convert data to DataFrame
df_features = pd.DataFrame(concept_data)

# Save results
df_features.to_csv("taxonomy_concept_features.csv", index=False)
print("Feature extraction complete! Saved as taxonomy_concept_features.csv")

# Display first few rows
import ace_tools_open as tools
tools.display_dataframe_to_user(name="Concept Features", dataframe=df_features)
