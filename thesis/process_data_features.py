import re
import networkx as nx
import numpy as np
from rdflib import Graph, Namespace
from rdflib.namespace import RDFS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from rapidfuzz import fuzz, distance
import pandas as pd

def extract_labels(graph):
    labels = {}
    for s, p, o in graph.triples((None, RDFS.label, None)):
        labels[str(s)] = [str(o).strip().lower()]
    for s, p, o in graph.triples((None, Namespace("http://purl.obolibrary.org/obo/").hasExactSynonym, None)):
        if str(s) in labels:
            labels[str(s)].append(str(o).strip().lower())
        else:
            labels[str(s)] = [str(o).strip().lower()]
    for s in list(labels.keys()):
        if len(labels[s]) == 0:
            labels[s] = [s]
    return labels

def label_pairs(pairs, exact_matches):
    exact_match_set = set(tuple(sorted([row["SrcEntity"], row["TgtEntity"]])) for row in exact_matches)
    labeled_pairs = []
    for c1, c2,_ in pairs:
        pair = tuple(sorted([c1, c2]))
        label = 1 if pair in exact_match_set else 0
        labeled_pairs.append((c1, c2, label))
    return labeled_pairs

def compute_features_ablation(source_labels,target_labels,labeled_pairs, concept_labels, concept_embeddings, hierarchy):
    G = nx.DiGraph()
    for rel in hierarchy:
        if rel["Predicate"] == "broader":
            G.add_edge(rel["Object"], rel["Subject"])
    concept_depths = {}
    for concept in nx.topological_sort(G):
        concept_depths[concept] = 0 if concept not in concept_depths else concept_depths[concept]
        for child in G.successors(concept):
            concept_depths[child] = concept_depths[concept] + 1
    source_concepts = [c for c in source_labels.keys()]
    target_concepts = [c for c in target_labels.keys()]
    max_depth_source = max([concept_depths.get(c, 0) for c in source_concepts])
    max_depth_target = max([concept_depths.get(c, 0) for c in target_concepts])

    # Function to compute ancestor set
    def ancestor_set(c):
        return set(nx.ancestors(G, c)) if c in G else set()

    # Function to compute normalized depth
    def norm_depth(c):
        d = concept_depths.get(c, 0)
        if c in source_labels:
            return d / max_depth_source if max_depth_source > 0 else 0
        elif c in target_labels:
            return d / max_depth_target if max_depth_target > 0 else 0
        else:
            return 0
    def shared_ancestors(c1, c2):
        return len(set(nx.ancestors(G, c1)) & set(nx.ancestors(G, c2))) if c1 in G and c2 in G else 0
    def lca_depth(c1, c2):
        common = set(nx.ancestors(G, c1)) & set(nx.ancestors(G, c2)) if c1 in G and c2 in G else set()
        return max([concept_depths[a] for a in common]) if common else 0

    data = []
    for c1, c2, label in labeled_pairs:
        l1 = normalize(' '.join(concept_labels.get(c1, [])))
        l2 = normalize(' '.join(concept_labels.get(c2, [])))
        e1, e2 = concept_embeddings.get(c1), concept_embeddings.get(c2)
        if e1 is None or e2 is None or np.isnan(e1).any() or np.isnan(e2).any():
            continue
        anc1 = ancestor_set(c1)
        anc2 = ancestor_set(c2)
        jaccard_anc = len(anc1 & anc2) / len(anc1 | anc2) if anc1 | anc2 else 0.0
        norm_depth_diff = abs(norm_depth(c1) - norm_depth(c2))
        features = {
            "Concept1": c1,
            "Concept2": c2,
            "Exact String Match": int(l1 == l2),
            "Partial String Match": fuzz.partial_ratio(l1, l2) / 100,
            "Weighted Jaccard": token_jaccard(l1, l2),
            "Cosine Similarity": cosine_sim(e1, e2),
            "Levenshtein Distance": distance.Levenshtein.distance(l1, l2),
            "Depth Difference": abs(concept_depths.get(c1, 0) - concept_depths.get(c2, 0)),
            # "Common Ancestor Depth": lca_depth(c1, c2),
            # "Shared Ancestor Count": shared_ancestors(c1, c2),
            "Norm Depth Difference": norm_depth_diff,
            "Ancestor Jaccard Similarity": jaccard_anc,
            "MatchLabel": label
        }
        data.append(features)
    return pd.DataFrame(data)

def extract_hierarchy(graph):
    hierarchy = []
    for s, p, o in graph.triples((None, RDFS.subClassOf, None)):
        hierarchy.append({"Subject": str(s), "Predicate": "broader", "Object": str(o)})
    return hierarchy

def cosine_sim(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0] if vec1 is not None and vec2 is not None else 0

def token_jaccard(a, b):
    set_a, set_b = set(a.split()), set(b.split())
    return len(set_a & set_b) / len(set_a | set_b) if set_a | set_b else 0.0

def compute_features(source_labels,target_labels,labeled_pairs, concept_labels, concept_embeddings, hierarchy):
    G = nx.DiGraph()
    for rel in hierarchy:
        if rel["Predicate"] == "broader":
            G.add_edge(rel["Object"], rel["Subject"])
    concept_depths = {}
    for concept in nx.topological_sort(G):
        concept_depths[concept] = 0 if concept not in concept_depths else concept_depths[concept]
        for child in G.successors(concept):
            concept_depths[child] = concept_depths[concept] + 1
    source_concepts = [c for c in source_labels.keys()]
    target_concepts = [c for c in target_labels.keys()]
    max_depth_source = max([concept_depths.get(c, 0) for c in source_concepts])
    max_depth_target = max([concept_depths.get(c, 0) for c in target_concepts])

    # Function to compute ancestor set
    def ancestor_set(c):
        return set(nx.ancestors(G, c)) if c in G else set()

    # Function to compute normalized depth
    def norm_depth(c):
        d = concept_depths.get(c, 0)
        if c in source_labels:
            return d / max_depth_source if max_depth_source > 0 else 0
        elif c in target_labels:
            return d / max_depth_target if max_depth_target > 0 else 0
        else:
            return 0
    def shared_ancestors(c1, c2):
        return len(set(nx.ancestors(G, c1)) & set(nx.ancestors(G, c2))) if c1 in G and c2 in G else 0
    def lca_depth(c1, c2):
        common = set(nx.ancestors(G, c1)) & set(nx.ancestors(G, c2)) if c1 in G and c2 in G else set()
        return max([concept_depths[a] for a in common]) if common else 0

    data = []
    for c1, c2, label in labeled_pairs:
        l1 = normalize(' '.join(concept_labels.get(c1, [])))
        l2 = normalize(' '.join(concept_labels.get(c2, [])))
        e1, e2 = concept_embeddings.get(c1), concept_embeddings.get(c2)
        if e1 is None or e2 is None or np.isnan(e1).any() or np.isnan(e2).any():
            continue
        anc1 = ancestor_set(c1)
        anc2 = ancestor_set(c2)
        jaccard_anc = len(anc1 & anc2) / len(anc1 | anc2) if anc1 | anc2 else 0.0
        norm_depth_diff = abs(norm_depth(c1) - norm_depth(c2))
        tokens1 = set(l1.split())
        tokens2 = set(l2.split())
        shared_tokens_count = len(tokens1 & tokens2)
        len_tokens1 = len(tokens1)
        len_tokens2 = len(tokens2)
        features = {
            "Concept1": c1,
            "Concept2": c2,
            "Shared Tokens Count": shared_tokens_count,
            "Token Ratio C1": shared_tokens_count / len_tokens1 if len_tokens1 > 0 else 0,
            "Token Ratio C2": shared_tokens_count / len_tokens2 if len_tokens2 > 0 else 0,
            "Exact String Match": int(l1 == l2),
            "Partial String Match": fuzz.partial_ratio(l1, l2) / 100,
            "Weighted Jaccard": token_jaccard(l1, l2),
            "Cosine Similarity": cosine_sim(e1, e2),
            "Levenshtein Distance": distance.Levenshtein.distance(l1, l2),
            "Depth Difference": abs(concept_depths.get(c1, 0) - concept_depths.get(c2, 0)),
            # "Common Ancestor Depth": lca_depth(c1, c2),
            # "Shared Ancestor Count": shared_ancestors(c1, c2),
            "Norm Depth Difference": norm_depth_diff,
            "Ancestor Jaccard Similarity": jaccard_anc,
            "MatchLabel": label
        }
        data.append(features)
    return pd.DataFrame(data)


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
