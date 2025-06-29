import spacy
import numpy as np
import pandas as pd
import random
from rdflib import Graph, Namespace
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import shap
import matplotlib.pyplot as plt
'''Random Forest is often preferred in scenarios where model interpretability is important—like in medical fields or areas where 
understanding the decision-making process is crucial. It’s robust against overfitting and generally performs well across a wide range of 
applications without the need for tuning.
XGBoost is often the algorithm of choice in machine learning competitions, such as those on Kaggle, where the highest possible accuracy 
is typically the goal. It excels in scenarios where the data is structured/tabular and the problem is sufficiently complex..'''
# Define Namespaces
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
SKOSXL = Namespace("http://www.w3.org/2008/05/skos-xl#")
RDF = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")

# Load RDF Graph
g = Graph()
g.parse("new_taxonomy_skos.xml", format="xml")

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
concept_data = {}

# Extract concepts and compute features
# for concept in g.subjects(predicate=RDF.type, object=SKOS.Concept):
#     # print(1)
#     label_resource = g.value(subject=concept, predicate=SKOSXL.prefLabel)
#     preferred_label = str(g.value(subject=label_resource, predicate=SKOSXL.literalForm)) if label_resource else None

#     if not preferred_label:
#         continue  # Skip if no label is available

#     # Collect alternate labels (synonyms)
#     synonyms = [str(syn) for syn in g.objects(subject=concept, predicate=SKOSXL.altLabel)]

#     # Extract cross-references (if available)
#     cross_refs = [str(ref) for ref in g.objects(subject=concept, predicate=SKOS.exactMatch)]
#     # if len(cross_refs)>0:
#     #     print('refs: ',len(cross_refs))

#     # Compute lexical similarity scores
#     label_embedding = nlp(preferred_label).vector if preferred_label else np.zeros(300)

#     lexical_features = {
#         "Concept": str(concept),
#         "Preferred Label": preferred_label,
#         "LexicalMatcher": 1 if any(preferred_label.lower() == syn.lower() for syn in synonyms) else 0,
#         "SpacelessLexicalMatcher": 1 if any(preferred_label.replace(" ", "").lower() == syn.replace(" ", "").lower() for syn in synonyms) else 0,
#         "WordMatcher": max(weighted_jaccard(preferred_label, syn) for syn in synonyms) if synonyms else 0,
#         "StringMatcher": max(fuzz.partial_ratio(preferred_label, syn) / 100 for syn in synonyms) if synonyms else 0,
#         "DirectXRefMatcher": 1 if cross_refs else 0,
#         "LLM Matcher": max(cosine_sim(label_embedding, nlp(str(syn)).vector) for syn in synonyms) if synonyms else 0
#     }

#     concept_data[str(concept)] = lexical_features
print(2)
# Convert to DataFrame
df_features = pd.read_csv("pairwise_concept_features.csv")
# df_features = pd.DataFrame(concept_data).T.reset_index(drop=True)

# Generate Training Data: Positive and Negative Pairs
positive_pairs = []
negative_pairs = []
#questa cosa è una follia computazionale, basterebbe mettere solo i concept che hanno un label exact match, che dovrebbe essere simmetrico
for concept1 in df_features["Concept"]:
    # print(3)
    # print('equals list',list(g.objects(subject=concept1, predicate=SKOS.exactMatch)))
    for concept2 in df_features["Concept"]:
        if concept1 != concept2:
            # Positive pairs: Exact matches
            if concept2 in list(g.objects(subject=concept1, predicate=SKOS.exactMatch)):
                print('exact match')
                positive_pairs.append((concept1, concept2, 1))
            else:
                negative_pairs.append((concept1, concept2, 0))

# Ensure balance between positive and negative pairs
negative_pairs = random.sample(negative_pairs, min(len(negative_pairs), len(positive_pairs)))
pairs = positive_pairs + negative_pairs
random.shuffle(pairs)

# Prepare feature vectors for model
X = []
y = []
print(4)
for c1, c2, label in pairs:
    if c1 in concept_data and c2 in concept_data:
        X.append(np.concatenate([list(concept_data[c1].values())[1:], list(concept_data[c2].values())[1:]]))
        y.append(label)

X = np.array(X)
y = np.array(y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate Model
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance using SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Convert SHAP values to feature importance
feature_names = list(concept_data[list(concept_data.keys())[0]].keys())[1:] * 2  # Duplicate for pairwise comparison
shap.summary_plot(shap_values[1], X_test, feature_names=feature_names)

# Display feature importance as DataFrame
feature_importance = pd.DataFrame({"Feature": feature_names, "Importance": np.abs(shap_values[1]).mean(axis=0)})
feature_importance = feature_importance.groupby("Feature").mean().sort_values("Importance", ascending=False)

# Display Feature Importance
import ace_tools as tools
tools.display_dataframe_to_user(name="Feature Importance", dataframe=feature_importance)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance.index, feature_importance["Importance"])
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance for Concept Mapping")
plt.gca().invert_yaxis()
plt.show()
