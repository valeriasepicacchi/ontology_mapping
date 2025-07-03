import spacy
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from rapidfuzz import fuzz, distance
import re
from sklearn.model_selection import train_test_split
import os
import random
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from generate_candidates import generate_candidate_pairs, generate_candidate_pairs_with_negatives_manual, deep_onto_generation
from process_data_features import normalize
import os
import pandas as pd
from render_output import create_table_result, create_figure
from process_data_features import token_jaccard, cosine_sim, extract_labels, extract_hierarchy, compute_features, compute_features_ablation
from rdflib import Graph


def main(param1, param2):
    # Load SpaCy model
    nlp = spacy.load("en_core_web_md")

    # load ontologies
    source_graph = Graph()
    target_graph = Graph()

    source_graph.parse("/Users/valeriasepicacchi/Documents/ontology_mapping/thesis/bio-ml/snomed-fma.body/snomed.body.owl", format="xml")
    target_graph.parse("/Users/valeriasepicacchi/Documents/ontology_mapping/thesis/bio-ml/snomed-fma.body/fma.body.owl", format="xml")

    source_labels = extract_labels(source_graph)
    target_labels = extract_labels(target_graph)
    concept_labels = {**source_labels, **target_labels}
    hierarchy = extract_hierarchy(source_graph) + extract_hierarchy(target_graph)

    # generate embeddings
    labels_to_embed = [' '.join(labels) for labels in concept_labels.values()]
    docs = list(nlp.pipe(labels_to_embed, disable=["ner", "parser"]))

    # load training data
    df_train_alignment = pd.read_csv("/Users/valeriasepicacchi/Documents/ontology_mapping/thesis/bio-ml/snomed-fma.body/refs_equiv/train.tsv", sep="\t")
    df_train_alignment['Label'] = df_train_alignment['Score'].apply(lambda x: 1 if x == 1.0 else 0)

    ref_train_mappings = df_train_alignment[df_train_alignment['Label'] == 1][['SrcEntity', 'TgtEntity']].to_dict('records')

    concept_embeddings = {uri: doc.vector for uri, doc in zip(concept_labels.keys(), docs)}

    df_test_alignment = pd.read_csv("/Users/valeriasepicacchi/Documents/ontology_mapping/thesis/bio-ml/snomed-fma.body/refs_equiv/test.tsv", sep="\t")
    df_test_alignment['Label'] = df_test_alignment['Score'].apply(lambda x: 1 if x == 1.0 else 0)
    ref_test_mappings = df_test_alignment[df_test_alignment['Label'] == 1][['SrcEntity', 'TgtEntity']].to_dict('records')

    # Generate training pairs
    train_pairs = generate_candidate_pairs_with_negatives_manual(
    ref_train_mappings,          
    source_labels,
    target_labels,
    hierarchy,
    idf_negatives=50,
    neighbour_negatives=50,
    boolean='training'
)
    # Generate testing pairs
    test_pairs = generate_candidate_pairs_with_negatives_manual(
    ref_test_mappings,           
    source_labels,
    target_labels,
    hierarchy,
    idf_negatives=50,
    neighbour_negatives=50,
    boolean= 'testing'
)

    # Compute features 
    features_train_df = compute_features(source_labels, target_labels, train_pairs, concept_labels, concept_embeddings, hierarchy)
    features_test_df  = compute_features(source_labels, target_labels, test_pairs, concept_labels, concept_embeddings, hierarchy)

    # Train classifier 
    X_train = features_train_df.drop(columns=["Concept1", "Concept2", "MatchLabel"]) # of course we drop the ground truth label
    y_train = features_train_df["MatchLabel"]

    X_test  = features_test_df.drop(columns=["Concept1", "Concept2", "MatchLabel"])
    y_test  = features_test_df["MatchLabel"]
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return clf, X_train, y_train, X_test, y_test, y_pred

param2 ="SNOMED-FMA"
for param1 in [[1,1,1],[1,1,2],[1,2,1],[2,1,1],[1,2,2],[2,1,2],[2,2,1]]:

    accumulated_report = defaultdict(lambda: defaultdict(list))
    accumulated_report['accuracy'] = [] 
    print(f"Running experiment on dataset: {param2}")
    for seed in [42, 56, 25, 111, 4]:
        random.seed(seed)
        clf, X_train, y_train, X_test, y_test, y_pred = main(param1,param2)
        report_dict = classification_report(y_test, y_pred, output_dict=True)
    
            # Accumulate results
        for key, value_dict in report_dict.items():
            if isinstance(value_dict, dict):  # skip accuracy as it's a scalar
                for metric, value in value_dict.items():
                    accumulated_report[key][metric].append(value)
            else:
                # For accuracy, which is a single float
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


    path = "/Users/valeriasepicacchi/Documents/ontology_mapping/thesis/results/final_results/voting/final_results.txt"
    # create_table_result(average_report,param1,param2,2 )
    create_table_result(average_report,param1,param2,'min lexical similarity', 'min cosine similarity', path )
    img_path = '/Users/sepicacchiv/Desktop/thesis/results/final_results/snomed_fma/partial_ratio_{param1}_cosine_{param2}.png'
    create_figure(clf,X_train, y_test,y_pred,img_path)




