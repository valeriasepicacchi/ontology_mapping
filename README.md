# 🧠 Ontology Mapping with Ensemble Learning  
**Improving Taxonomy Mapping Performance with a Complex Ensemble Algorithm**

This repository contains the code and experiments for the Master's thesis:

> **"Improving taxonomy mapping performance with a complex ensemble algorithm"**  
> _Valeria Sepicacchi - July 2025_  
> Supervised by **Paul Groth**, mentored by **Wytze Vlietstra**

## 📘 Thesis Summary

Ontology and taxonomy alignment is crucial for ensuring semantic interoperability across biomedical vocabularies. This thesis proposes a **supervised ensemble approach** that combines lexical, structural, and semantic features to improve equivalence mapping between concepts in heterogeneous biomedical ontologies.

The proposed pipeline:
- Extracts concept labels and embeddings
- Generates candidate pairs using a hybrid similarity strategy
- Builds features (e.g., lexical, embedding-based, hierarchical)
- Trains an **ensemble classifier** (Voting: Logistic Regression, Gradient Boosting, Random Forest)
- Evaluates performance against state-of-the-art systems (BERTMap, LogMap, Matcha-DL, etc.)

The method is evaluated on the **Bio-ML track datasets** from the [Ontology Alignment Evaluation Initiative (OAEI)](http://oaei.ontologymatching.org/).

## 📊 Datasets

Three biomedical ontology pairs are used:

- **NCIT–DOID**: Cancer vs. general disease ontologies  
- **OMIM–ORDO**: Genetic vs. rare disease ontologies  
- **SNOMED–FMA**: Clinical terminology vs. anatomy

