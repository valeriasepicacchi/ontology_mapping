import spacy
import numpy as np
import pandas as pd
import networkx as nx
from rdflib import Graph, Namespace
from rdflib.namespace import RDFS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from rapidfuzz import fuzz, distance
import re
from sklearn.model_selection import train_test_split
import os
import random
from collections import defaultdict

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# --- Step 3: Extract Labels ---
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



# --- Step 4: Extract Hierarchy ---
def extract_hierarchy(graph):
    hierarchy = []
    for s, p, o in graph.triples((None, RDFS.subClassOf, None)):
        hierarchy.append({"Subject": str(s), "Predicate": "broader", "Object": str(o)})
    return hierarchy



# --- Step 6: Generate Candidate Pairs ---
def generate_candidate_pairs(source_labels, target_labels, concept_labels, concept_embeddings, top_k, min_lexical_sim, min_cosine_sim):
    source_uris = list(source_labels.keys())
    target_uris = list(target_labels.keys())
    source_vectors = np.array([concept_embeddings[uri] for uri in source_uris])
    target_vectors = np.array([concept_embeddings[uri] for uri in target_uris])
    similarity_matrix = cosine_similarity(source_vectors, target_vectors)
    candidate_pairs = set()
    for i, src_uri in enumerate(source_uris):
        sims = similarity_matrix[i]
        top_indices = np.argsort(sims)[::-1][:top_k]
        for j in top_indices:
            tgt_uri = target_uris[j]
            cos_sim = sims[j]
            if cos_sim < min_cosine_sim:
                continue
            label1 = normalize(' '.join(concept_labels.get(src_uri, [])))
            label2 = normalize(' '.join(concept_labels.get(tgt_uri, [])))
            lexical_sim = fuzz.partial_ratio(label1, label2)
            if lexical_sim >= min_lexical_sim:
                candidate_pairs.add((src_uri, tgt_uri))
    return list(candidate_pairs)

# --- Step 7: Label Candidate Pairs with Capped Negatives ---
def label_candidate_pairs(candidate_pairs, exact_matches, max_neg_per_pos=1):
    exact_match_set = set(tuple(sorted([row["SrcEntity"], row["TgtEntity"]])) for row in exact_matches)
    candidate_pairs = [tuple(sorted([c1, c2])) for c1, c2 in candidate_pairs]
    positives = []
    negatives = []
    for c1, c2 in candidate_pairs:
        pair = tuple(sorted([c1, c2]))
        label = 1 if pair in exact_match_set else 0
        if label == 1:
            positives.append((c1, c2, label))
        else:
            negatives.append((c1, c2, label))
    max_neg_per_pos = 2
    max_negatives = int(round(min(len(negatives), max_neg_per_pos * float(len(positives)))))
    negatives = random.sample(negatives, k=max_negatives)
    # negatives = list(np.random.choice(negatives, size=max_negatives, replace=False))
    labeled_pairs = positives + negatives
    np.random.shuffle(labeled_pairs)
    return labeled_pairs

# --- Step 8: Feature Computation ---
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
def create_table_result(report_dict,min_lexical_sim,min_cosine_sim,max_neg_per_pos ):
    # --- Classification Report ---
    

    # --- Define your "header" for the LaTeX table ---
    latex_lines = []
    latex_lines.append("\\begin{tabular}{ |p{2cm}||p{2cm}|p{2cm}|p{2cm}|p{2cm}|  }")
    latex_lines.append(" \\hline")

    # You can customize this with your current parameters:
    table_title = f"max negatives = {max_neg_per_pos}, fuzz partial ratio > {min_lexical_sim}, cosine similarity > {min_cosine_sim}"
    latex_lines.append(f" \\multicolumn{{5}}{{|c|}}{{{table_title}}} \\\\")
    latex_lines.append(" \\hline")

    latex_lines.append("  & Precision & Recall & F1 score & support\\\\")
    latex_lines.append(" \\hline")

    # --- Add class 0 ---
    cls_0 = report_dict["0"]
    latex_lines.append(
        f"0   &   {cls_0['precision']:.2f}  &    {cls_0['recall']:.2f}   &   {cls_0['f1-score']:.2f}    &   {int(cls_0['support'])}\\\\"
    )
    latex_lines.append(" \\hline")

    # --- Add class 1 ---
    cls_1 = report_dict["1"]
    latex_lines.append(
        f"1   &   {cls_1['precision']:.2f}  &    {cls_1['recall']:.2f}   &   {cls_1['f1-score']:.2f}    &   {int(cls_1['support'])}\\\\"
    )
    latex_lines.append(" \\hline")

    # --- Add accuracy ---
    accuracy = report_dict["accuracy"]
    total_support = int(cls_0["support"] + cls_1["support"])
    latex_lines.append(
        f"accuracy & & &    {accuracy:.2f}    &   {total_support}\\\\"
    )
    latex_lines.append(" \\hline")

    # --- Add macro avg ---
    macro_avg = report_dict["macro avg"]
    latex_lines.append(
        f"macro-avg &   {macro_avg['precision']:.2f}   &   {macro_avg['recall']:.2f}  &    {macro_avg['f1-score']:.2f}    &   {int(macro_avg['support'])}\\\\"
    )
    latex_lines.append(" \\hline")

    # --- Add weighted avg ---
    weighted_avg = report_dict["weighted avg"]
    latex_lines.append(
        f"weighted avg &   {weighted_avg['precision']:.2f}   &   {weighted_avg['recall']:.2f}  &    {weighted_avg['f1-score']:.2f}    &   {int(weighted_avg['support'])}\\\\"
    )
    latex_lines.append(" \\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\\\")
    latex_lines.append("\\\\")

    # --- Append to text file ---
    with open("results/classification_report_latex.txt", "a") as f:  # << HERE: "a" means append
        f.write("\n".join(latex_lines))
        f.write("\n\n")  # Add some spacing between tables

def generate_all_possible_pairs(source_labels, target_labels):
    source_uris = list(source_labels.keys())
    target_uris = list(target_labels.keys())
    all_pairs = [(src_uri, tgt_uri) for src_uri in source_uris for tgt_uri in target_uris]
    return all_pairs

def select_training_pairs_fast(train_pairs, concept_labels, concept_embeddings, source_uris, target_uris, similarity_matrix, min_lexical_sim, min_cosine_sim, exact_match_set):
    selected_pairs = []
    source_index = {uri: i for i, uri in enumerate(source_uris)}
    target_index = {uri: i for i, uri in enumerate(target_uris)}

    for c1, c2, label in train_pairs:
        i = source_index.get(c1)
        j = target_index.get(c2)
        if i is None or j is None:
            continue  # skip if not found

        cos_sim = similarity_matrix[i, j]
        if cos_sim < min_cosine_sim:
            continue

        l1 = normalize(' '.join(concept_labels.get(c1, [])))
        l2 = normalize(' '.join(concept_labels.get(c2, [])))
        lexical_sim = fuzz.partial_ratio(l1, l2)

        if lexical_sim >= min_lexical_sim:
            selected_pairs.append((c1, c2, label))

    return selected_pairs
# Label pairs with 1/0 depending on alignment
def label_pairs(pairs, exact_matches):
    exact_match_set = set(tuple(sorted([row["SrcEntity"], row["TgtEntity"]])) for row in exact_matches)
    labeled_pairs = []
    for c1, c2 in pairs:
        pair = tuple(sorted([c1, c2]))
        label = 1 if pair in exact_match_set else 0
        labeled_pairs.append((c1, c2, label))
    return labeled_pairs

# Select training pairs based on your thresholds
def select_training_pairs(train_pairs, concept_labels, concept_embeddings, min_lexical_sim, min_cosine_sim):
    selected_pairs = []
    for c1, c2, label in train_pairs:
        l1 = normalize(' '.join(concept_labels.get(c1, [])))
        l2 = normalize(' '.join(concept_labels.get(c2, [])))
        lexical_sim = fuzz.partial_ratio(l1, l2)
        cos_sim = cosine_sim(concept_embeddings.get(c1), concept_embeddings.get(c2))
        if lexical_sim >= min_lexical_sim and cos_sim >= min_cosine_sim:
            selected_pairs.append((c1, c2, label))
    return selected_pairs

def main(param1, param2):
    # Load SpaCy model
    nlp = spacy.load("en_core_web_md")

    # --- Step 1: Load Ontologies ---
    source_graph = Graph()
    target_graph = Graph()

    source_graph.parse("/Users/sepicacchiv/Downloads/ncit-doid/ncit.owl", format="xml")
    target_graph.parse("/Users/sepicacchiv/Desktop/thesis/MONDO/equiv_match/ontos/ordo.owl", format="xml")

    # --- Step 2: Load Reference Alignment (TSV) ---
    df_alignment = pd.read_csv("/Users/sepicacchiv/Desktop/thesis/MONDO/equiv_match/refs/omim2ordo/full.tsv", sep="\t")
    df_alignment['Label'] = df_alignment['Score'].apply(lambda x: 1 if x == 1.0 else 0)

    source_labels = extract_labels(source_graph)
    target_labels = extract_labels(target_graph)
    concept_labels = {**source_labels, **target_labels}
    hierarchy = extract_hierarchy(source_graph) + extract_hierarchy(target_graph)

    # --- Step 5: Generate Concept Embeddings ---
    labels_to_embed = [' '.join(labels) for labels in concept_labels.values()]
    docs = list(nlp.pipe(labels_to_embed, disable=["ner", "parser"]))
    concept_embeddings = {uri: doc.vector for uri, doc in zip(concept_labels.keys(), docs)}

    # --- Step 6: Generate ALL possible pairs ---
    all_pairs = generate_all_possible_pairs(source_labels, target_labels)
    labeled_all_pairs = label_pairs(all_pairs, df_alignment.to_dict('records'))

    # --- Step 7: Split pairs into train/test ---
    random.seed(42)
    random.shuffle(labeled_all_pairs)
    split_ratio = 0.8
    split_idx = int(len(labeled_all_pairs) * split_ratio)

    train_pairs = labeled_all_pairs[:split_idx]
    test_pairs = labeled_all_pairs[split_idx:]

    # --- Precompute cosine similarity matrix ONCE ---
    source_uris = list(source_labels.keys())
    target_uris = list(target_labels.keys())
    source_vectors = np.array([concept_embeddings[uri] for uri in source_uris])
    target_vectors = np.array([concept_embeddings[uri] for uri in target_uris])
    similarity_matrix = cosine_similarity(source_vectors, target_vectors)

    # --- Prepare exact_match_set ONCE ---
    exact_match_set = set(tuple(sorted([row["SrcEntity"], row["TgtEntity"]])) for row in df_alignment.to_dict('records'))

    # --- Step 8: Select training pairs based on thresholds (FAST) ---
    selected_train_pairs = select_training_pairs_fast(
        train_pairs, concept_labels, concept_embeddings,
        source_uris, target_uris, similarity_matrix,
        param1, param2, exact_match_set
    )

    # --- Step 9: Compute features ---
    features_train_df = compute_features(source_labels, target_labels, selected_train_pairs, concept_labels, concept_embeddings, hierarchy)
    features_test_df  = compute_features(source_labels, target_labels, test_pairs, concept_labels, concept_embeddings, hierarchy)

    # --- Step 10: Train classifier ---
    X_train = features_train_df.drop(columns=["Concept1", "Concept2", "MatchLabel"])
    y_train = features_train_df["MatchLabel"]

    X_test  = features_test_df.drop(columns=["Concept1", "Concept2", "MatchLabel"])
    y_test  = features_test_df["MatchLabel"]

    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # Suppose you know the total number of reference positives:
    total_reference_positives = df_alignment[df_alignment['Label'] == 1].shape[0]

    # Approximate number of positives that would fall in test set:
    approx_test_positives = int(0.25 * total_reference_positives)

    # Now compare:
    test_predicted_positives_indices = np.where((y_test == 1) & (y_pred == 1))[0]
    test_predicted_positives_count = len(test_predicted_positives_indices)

    # Approximate recall:
    approx_recall = test_predicted_positives_count / approx_test_positives if approx_test_positives > 0 else 0

    print(f"[APPROX TEST RECALL] Total reference positives: {total_reference_positives}")
    print(f"[APPROX TEST RECALL] Estimated positives in test set: {approx_test_positives}")
    print(f"[APPROX TEST RECALL] True positives predicted in test set: {test_predicted_positives_count}")
    print(f"[APPROX TEST RECALL] Approximate recall = {approx_recall:.4f}")
    return clf, X_train, y_train, X_test, y_test, y_pred


def create_figure(clf,X_train, y_test,y_pred):
    from collections import Counter
    # print("[DEBUG] Label distribution:", Counter(y))

    # --- Feature Importances ---
    # importances = clf.feature_importances_
    # plt.figure(figsize=(10, 5))
    # plt.barh(X_train.columns, importances)
    # plt.title("Feature Importances")
    # plt.xlabel("Importance")
    # plt.ylabel("Feature")
    # plt.tight_layout()
    # plt.show()

    # os.makedirs("results", exist_ok=True)

    # --- Feature Importances ---
    importances = clf.feature_importances_

    plt.figure(figsize=(10, 5))
    plt.barh(X_train.columns, importances)
    plt.title("Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()

    # --- Save figure ---
    plt.savefig(f"results/partial_ratio_{param1}_cosine_{param2}.png", dpi=300)
    # create_table_result(y_test,y_pred,param1,param2,2 )

for param1 in range(50, 61, 10):    # 0, 10, 20, ..., 100
    for param2 in np.arange(0.5, 0.61, 0.1):
        accumulated_report = defaultdict(lambda: defaultdict(list))
        accumulated_report['accuracy'] = [] 
        print(f"Running experiment with partial ratio {param1} and cosine similarity {param2}")
        for seed in [42]:
            random.seed(seed)
            clf, X_train, y_train, X_test, y_test, y_pred = main(param1,param2)
            report_dict = classification_report(y_test, y_pred, output_dict=True)
        
                # Accumulate results
            for key, value_dict in report_dict.items():
                if isinstance(value_dict, dict):  # skip accuracy as it's a scalar
                    for metric, value in value_dict.items():
                        accumulated_report[key][metric].append(value)
                else:
                    # For 'accuracy', which is a single float
                    accumulated_report['accuracy'].append(value_dict)

        average_report = {}

        for key, value_dict in accumulated_report.items():
            if key == 'accuracy':
                average_report[key] = np.mean(value_dict)  # value_dict is a list of floats
                # average_report[key] = average
            else:
                average_report[key] = {}
                for metric, value_list in value_dict.items():
                    average = np.mean(value_list)
                    average_report[key][metric] = average

        create_table_result(average_report,param1,param2,2 )
        create_figure(clf,X_train, y_test,y_pred)




