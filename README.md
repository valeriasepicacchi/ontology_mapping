# Ontology Mapping with Ensemble Learning  
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

## ⚙️ Environment Setup

This project uses a Conda environment for reproducibility.

To create and activate the environment:

```bash
conda env create -f thesis/environment.yml
conda activate myenv
```
## Results
The results for the voting ensamble on the three datasets can be replicated running the main script. The main script outputs results in the shape of tables, that are saved in thesis/results/final_results/voting/final_results.txt 
Hence, the numerical results for the training on the three datasets with different weights can be found in thesis/results/final_results/voting/final_results.txt

The folder thesis/results is further split in folders, one for each dataset, that show the results for the experiments with feature. It's also included the feature importance bar plot for every dataset and the results for the ablation study.