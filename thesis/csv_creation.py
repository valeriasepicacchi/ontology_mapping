import uuid
import random
import spacy
import numpy as np
import pandas as pd
import networkx as nx
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF
from rapidfuzz import fuzz, distance
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load SpaCy model
nlp = spacy.load("en_core_web_md")

# Define Namespaces
skos = Namespace("http://www.w3.org/2004/02/skos/core#")
skosxl = Namespace("http://www.w3.org/2008/05/skos-xl#")
DCT = Namespace("http://purl.org/dc/terms/")

def label_candidate_pairs(candidate_pairs, exact_matches):
    exact_match_set = set()
    for match in exact_matches:
        c1, c2 = match["Concept1"], match["Concept2"]
        exact_match_set.add(tuple(sorted([c1, c2])))

    labeled_pairs = []
    for c1, c2 in candidate_pairs:
        label = 1 if tuple(sorted([c1, c2])) in exact_match_set else 0
        labeled_pairs.append((c1, c2, label))
    
    return labeled_pairs

def compute_pairwise_features_labeled(concept_labels, concept_embeddings, hierarchy, labeled_pairs):
    pairwise_data = []

    G = nx.DiGraph()
    for rel in hierarchy:
        if rel["Predicate"] == "broader":
            G.add_edge(rel["Object"], rel["Subject"])

    concept_depths = {}
    for concept in nx.topological_sort(G):
        if concept not in concept_depths:
            concept_depths[concept] = 0
        for child in G.successors(concept):
            concept_depths[child] = concept_depths[concept] + 1

    def lca_depth(c1, c2):
        try:
            ancestors1 = set(nx.ancestors(G, c1))
            ancestors2 = set(nx.ancestors(G, c2))
            common = ancestors1 & ancestors2
            return max([concept_depths[a] for a in common]) if common else 0
        except:
            return 0

    def shared_ancestors(c1, c2):
        try:
            return len(set(nx.ancestors(G, c1)) & set(nx.ancestors(G, c2)))
        except:
            return 0

    def overlap_ratio(c1, c2):
        shared = shared_ancestors(c1, c2)
        depth = max(concept_depths.get(c1, 0), concept_depths.get(c2, 0))
        return shared / depth if depth > 0 else 0

    for c1, c2, label in labeled_pairs:
        l1, l2 = concept_labels[c1], concept_labels[c2]
        e1, e2 = concept_embeddings[c1], concept_embeddings[c2]

        if c1 not in G or c2 not in G:
            dist = -1
        else:
            try:
                dist = nx.shortest_path_length(G, source=c1, target=c2)
            except nx.NetworkXNoPath:
                dist = -1

        features = {
            "Concept1": c1,
            "Concept2": c2,
            "Exact String Match": int(l1 == l2),
            "Partial String Match": fuzz.partial_ratio(l1, l2) / 100,
            "Weighted Jaccard": token_jaccard(l1, l2),
            "Cosine Similarity": cosine_sim(e1, e2),
            "Levenshtein Distance": distance.Levenshtein.distance(l1, l2),
            "Depth Difference": abs(concept_depths.get(c1, 0) - concept_depths.get(c2, 0)),
            "Common Ancestor Depth": lca_depth(c1, c2),
            "Shared Ancestor Count": shared_ancestors(c1, c2),
            "Hierarchical Overlap Ratio": overlap_ratio(c1, c2),
            "Hierarchical Distance": dist,
            "MatchLabel": label
        }

        pairwise_data.append(features)

    return pairwise_data

# Utility functions
def normalize_label(label):
    return label.lower().strip() if label else ""

def cosine_sim(vec1, vec2):
    if vec1 is None or vec2 is None:
        return 0
    vec1, vec2 = np.array(vec1).reshape(1, -1), np.array(vec2).reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]

def token_jaccard(a, b):
    set_a, set_b = set(a.split()), set(b.split())
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0

def create_taxonomy_data(g):
    concepts = []
    hierarchy = []
    exact_matches = []
    concept_labels = {}
    concept_embeddings = {}

    c = 0
    concept_uris = list(g.subjects(predicate=RDF.type, object=skos.Concept))

    for concept in concept_uris:
        label_resource = g.value(subject=concept, predicate=skosxl.prefLabel)
        preferred_label = g.value(subject=label_resource, predicate=skosxl.literalForm) if label_resource else None
        preferred_label = str(preferred_label) if preferred_label else ""
        norm_label = normalize_label(preferred_label)
        label_resource_alt = g.value(subject=concept, predicate=skosxl.altLabel)
        alt_label = g.value(subject=label_resource_alt, predicate=skosxl.literalForm) if label_resource_alt else None
        norm_label_alt = normalize_label(alt_label)
        if c % 3 != 0:
            uri = str(concept)
            concepts.append({"ConceptURI": uri, "PreferredLabel": preferred_label, "LabelType": "original"})
            concept_labels[uri] = norm_label

            for b in g.objects(subject=concept, predicate=skos.broader):
                hierarchy.append({"Subject": uri, "Predicate": "broader", "Object": str(b)})
            for n in g.objects(subject=concept, predicate=skos.narrower):
                hierarchy.append({"Subject": uri, "Predicate": "narrower", "Object": str(n)})

        else:
            uri1 = str(concept) + "1"
            uri2 = str(concept) + "2"

            concepts.append({"ConceptURI": uri1, "PreferredLabel": preferred_label, "LabelType": "split_1"})
            concepts.append({"ConceptURI": uri2, "PreferredLabel": alt_label, "LabelType": "split_2"})

            concept_labels[uri1] = norm_label
            concept_labels[uri2] = norm_label_alt

            exact_matches.append({"Concept1": uri1, "Concept2": uri2, "Relation": "exactMatch"})
            exact_matches.append({"Concept1": uri2, "Concept2": uri1, "Relation": "exactMatch"})

            for b in g.objects(subject=concept, predicate=skos.broader):
                hierarchy.append({"Subject": uri1, "Predicate": "broader", "Object": str(b)})
                hierarchy.append({"Subject": uri2, "Predicate": "broader", "Object": str(b)})
            for n in g.objects(subject=concept, predicate=skos.narrower):
                hierarchy.append({"Subject": uri1, "Predicate": "narrower", "Object": str(n)})
                hierarchy.append({"Subject": uri2, "Predicate": "narrower", "Object": str(n)})

        c += 1
    print('totali concetti sono', c)
    # Batch embedding
    labels_to_embed = list(concept_labels.values())
    docs = list(nlp.pipe(labels_to_embed, disable=["ner", "parser"]))
    for i, uri in enumerate(concept_labels.keys()):
        concept_embeddings[uri] = docs[i].vector

    return concepts, hierarchy, exact_matches, concept_labels, concept_embeddings

# def compute_pairwise_features(concept_labels, concept_embeddings, hierarchy):
#     concepts = list(concept_labels.keys())
#     pairwise_data = []

#     G = nx.DiGraph()
#     for rel in hierarchy:
#         if rel["Predicate"] == "broader":
#             G.add_edge(rel["Object"], rel["Subject"])

#     concept_depths = {}
#     for concept in nx.topological_sort(G):
#         if concept not in concept_depths:
#             concept_depths[concept] = 0
#         for child in G.successors(concept):
#             concept_depths[child] = concept_depths[concept] + 1

#     def lca_depth(c1, c2):
#         try:
#             ancestors1 = set(nx.ancestors(G, c1))
#             ancestors2 = set(nx.ancestors(G, c2))
#             common = ancestors1 & ancestors2
#             return max([concept_depths[a] for a in common]) if common else 0
#         except:
#             return 0

#     def shared_ancestors(c1, c2):
#         try:
#             return len(set(nx.ancestors(G, c1)) & set(nx.ancestors(G, c2)))
#         except:
#             return 0

#     def overlap_ratio(c1, c2):
#         shared = shared_ancestors(c1, c2)
#         depth = max(concept_depths.get(c1, 0), concept_depths.get(c2, 0))
#         return shared / depth if depth > 0 else 0

#     for i, c1 in enumerate(concepts):
#         for j, c2 in enumerate(concepts):
#             if i > j:
#                 continue

#             l1, l2 = concept_labels[c1], concept_labels[c2]
#             e1, e2 = concept_embeddings[c1], concept_embeddings[c2]

#             try:
#                 dist = nx.shortest_path_length(G, source=c1, target=c2)
#             except nx.NetworkXNoPath:
#                 dist = -1

#             features = {
#                 "Concept1": c1,
#                 "Concept2": c2,
#                 #"Exact String Match": int(l1 == l2),
#                 "Partial String Match": fuzz.partial_ratio(l1, l2) / 100,
#                 "Weighted Jaccard": token_jaccard(l1, l2),
#                 "Cosine Similarity": cosine_sim(e1, e2),
#                 "Levenshtein Distance": distance.Levenshtein.distance(l1, l2),
#                 "Depth Difference": abs(concept_depths.get(c1, 0) - concept_depths.get(c2, 0)),
#                 "Common Ancestor Depth": lca_depth(c1, c2),
#                 "Shared Ancestor Count": shared_ancestors(c1, c2),
#                 "Hierarchical Overlap Ratio": overlap_ratio(c1, c2),
#                 "Hierarchical Distance": dist
#             }

#             pairwise_data.append(features)
#         if i >= 100:
#             break

#     return pairwise_data
def generate_candidate_pairs(concept_labels, concept_embeddings, top_k=100, min_lexical_sim=30, min_cosine_sim=0.3):
    from sklearn.metrics.pairwise import cosine_similarity

    uris = list(concept_labels.keys())
    vectors = np.array([concept_embeddings[uri] for uri in uris])
    similarity_matrix = cosine_similarity(vectors)

    candidate_pairs = set()

    for i, uri1 in enumerate(uris):
        sims = similarity_matrix[i]
        top_indices = np.argsort(sims)[::-1][1:top_k+1]  # skip self

        for j in top_indices:
            uri2 = uris[j]
            cos_sim = sims[j]
            if cos_sim < min_cosine_sim:
                continue

            label1 = concept_labels[uri1]
            label2 = concept_labels[uri2]
            lexical_sim = fuzz.partial_ratio(label1, label2)

            if lexical_sim >= min_lexical_sim:
                pair = tuple(sorted([uri1, uri2]))
                candidate_pairs.add(pair)

    return list(candidate_pairs)

# Main execution
# g = Graph()
# g.parse("emtree_release_202501.xml", format="xml")

# concepts, hierarchy, exact_matches, labels, embeddings = create_taxonomy_data(g)
# print('features calculated')
# pd.DataFrame(concepts).to_csv("concepts.csv", index=False)
# pd.DataFrame(hierarchy).to_csv("hierarchy.csv", index=False)
# pd.DataFrame(exact_matches).to_csv("exact_matches.csv", index=False)


# pairwise_data = compute_pairwise_features(labels, embeddings, hierarchy)
# df_features = pd.DataFrame(pairwise_data)
# df_features.to_csv("pairwise_concept_features.csv", index=False)

# # Correlation heatmap
# # plt.figure(figsize=(10, 6))
# plt.figure(figsize=(10,6))
# numeric_df = df_features.select_dtypes(include=[np.number])
# sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
# # sns.heatmap(df_features.corr(), annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Feature Correlation Matrix")
# plt.show()
g = Graph()
g.parse("emtree_release_202501.xml", format="xml")

concepts, hierarchy, exact_matches, concept_labels, concept_embeddings = create_taxonomy_data(g)
candidate_pairs = generate_candidate_pairs(concept_labels, concept_embeddings, top_k=20)
labeled_pairs = label_candidate_pairs(candidate_pairs, exact_matches)

# Compute features
pairwise_data = compute_pairwise_features_labeled(
    concept_labels,
    concept_embeddings,
    hierarchy,
    labeled_pairs
)

df = pd.DataFrame(pairwise_data)

# Prepare data for training
X = df.drop(columns=["Concept1", "Concept2", "MatchLabel"])
y = df["MatchLabel"]

# print(X.describe())  # Summary stats per column
# print("Min:", X.min().min())
# print("Max:", X.max().max())
# print("Any infs?", np.isinf(X.to_numpy()).any())
# print("Any NaNs?", X.isna().any().any())

# X = X.replace([np.inf, -np.inf], np.nan)
# X = X.fillna(-1)
# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
from sklearn.utils import resample

# Combine X and y for easier sampling
df_train = pd.DataFrame(X, columns=X.columns)
df_train['MatchLabel'] = y.values

# Separate classes
df_pos = df_train[df_train['MatchLabel'] == 1]
df_neg = df_train[df_train['MatchLabel'] == 0]

# Downsample negatives
df_neg_downsampled = resample(df_neg, replace=False, n_samples=len(df_pos) * 5, random_state=42)

# Combine and shuffle
df_balanced = pd.concat([df_pos, df_neg_downsampled]).sample(frac=1, random_state=42)
X_train_bal = df_balanced.drop(columns='MatchLabel')
y_train_bal = df_balanced['MatchLabel']
X_train, X_test, y_train, y_test = train_test_split(X_train_bal, y_train_bal, stratify=y_train_bal, test_size=0.25, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Plot feature importance
importances = clf.feature_importances_
features = X.columns
plt.figure(figsize=(10, 5))
plt.barh(features, importances)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()
