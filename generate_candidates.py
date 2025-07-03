from deeponto.onto import Ontology
from deeponto.align.mapping import NegativeCandidateMappingGenerator, ReferenceMapping
from deeponto.align.bertmap import BERTMapPipeline
from transformers import AutoTokenizer
import numpy as np
from process_data_features import normalize
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz, distance
from collections import defaultdict
import random 

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

def generate_candidate_pairs_with_negatives_manual(
    ref_alignment_df,
    source_labels,
    target_labels,
    hierarchy,
    idf_negatives,
    neighbour_negatives,
    boolean
):
    # Build hierarchy graph
    G = nx.DiGraph()
    for rel in hierarchy:
        if rel["Predicate"] == "broader":
            G.add_edge(rel["Object"], rel["Subject"])  # edge from parent → child

    # Prepare ancestors lookup
    def ancestor_set(c):
        return set(nx.ancestors(G, c)) if c in G else set()

    # Build inverted index for target labels
    inverted_index = defaultdict(set)
    for tgt_uri, labels in target_labels.items():
        text = normalize(' '.join(labels))
        for word in text.split():
            inverted_index[word].add(tgt_uri)

    # Prepare reference mappings set for easy lookup
    ref_pairs = set((row["SrcEntity"], row["TgtEntity"]) for row in ref_alignment_df)

    # For each positive pair → generate negatives
    labeled_pairs = []

    for i, row in enumerate(ref_alignment_df):
        src_uri = row["SrcEntity"]
        tgt_uri = row["TgtEntity"]

        # Add positive pair
        labeled_pairs.append((src_uri, tgt_uri, 1))

        # --- IDF Negatives (textually similar) ---
        src_text = normalize(' '.join(source_labels.get(src_uri, [])))
        candidate_tgt_uris = set()
        for word in src_text.split():
            candidate_tgt_uris.update(inverted_index.get(word, set()))

        candidate_tgt_uris.discard(tgt_uri)  # remove true match

        # Sample IDF negatives
        idf_neg_samples = random.sample(candidate_tgt_uris, min(idf_negatives, len(candidate_tgt_uris)))

        for neg_tgt_uri in idf_neg_samples:
            if (src_uri, neg_tgt_uri) in ref_pairs:
                continue
            labeled_pairs.append((src_uri, neg_tgt_uri, 0))

        # --- Neighbour Negatives (siblings) ---
        # Find siblings of tgt_uri → same parent
        siblings = set()
        parents = list(G.predecessors(tgt_uri))
        for parent in parents:
            siblings.update(G.successors(parent))
        siblings.discard(tgt_uri)  # remove true match
        siblings = [sib for sib in siblings if sib not in ancestor_set(tgt_uri)]  # exclude ancestors

        # Sample Neighbour negatives
        neighbour_neg_samples = random.sample(siblings, min(neighbour_negatives, len(siblings)))

        for neg_tgt_uri in neighbour_neg_samples:
            if (src_uri, neg_tgt_uri) in ref_pairs:
                continue
            labeled_pairs.append((src_uri, neg_tgt_uri, 0))

        # print(f"[{i+1}/{len(ref_alignment_df)}] PosPair ({src_uri}, {tgt_uri}) → {len(idf_neg_samples)} idf-negs, {len(neighbour_neg_samples)} neighbour-negs")

    print(f"[DONE] Total candidate pairs generated: {len(labeled_pairs)} for {boolean}")
    return labeled_pairs

def deep_onto_generation(
    source_owl_path,
    target_owl_path,
    train_or_test_tsv_path,
    idf_negatives,
    neighbour_negatives,
    max_hops=5,
    pretrained_bert_path="bert-base-uncased"
):
    """Generate dataset with positives and negatives, returning a list of (src_uri, tgt_uri, label) tuples."""



    # Load ontologies
    source_onto = Ontology(source_owl_path)
    target_onto = Ontology(target_owl_path)

    # Load reference mappings from TSV
    ref_mappings = ReferenceMapping.read_table_mappings(train_or_test_tsv_path)

    # Load BERTMap config
    config = BERTMapPipeline.load_bertmap_config()

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(getattr(config.bert, "pretrained_path", pretrained_bert_path))

    # Initialize the NegativeCandidateMappingGenerator
    cand_generator = NegativeCandidateMappingGenerator(
        source_onto,
        target_onto,
        ref_mappings,
        annotation_property_iris=config.annotation_property_iris,
        tokenizer=tokenizer,
        max_hops=max_hops,
        for_subsumptions=False
    )

    # For each positive pair → generate negatives
    pairs = []

    for i, pos_map in enumerate(ref_mappings):
        src_uri = pos_map.head
        tgt_uri = pos_map.tail

        # Generate negatives
        valid_negatives, stats = cand_generator.mixed_sample(pos_map, idf=idf_negatives, neighbour=neighbour_negatives)

        print(f"[INFO] ({i+1}/{len(ref_mappings)}) PosPair = ({src_uri}, {tgt_uri}) -> {len(valid_negatives)} negatives")

        # Add positive pair
        pairs.append((src_uri, tgt_uri, 1))

        # Add negative pairs
        for neg_tgt in valid_negatives:
            pairs.append((src_uri, neg_tgt, 0))

    print(f"[DONE] Generated total {len(pairs)} pairs ({len(ref_mappings)} positives + negatives)")

    return pairs

def generate_all_possible_pairs(source_labels, target_labels):
    source_uris = list(source_labels.keys())
    target_uris = list(target_labels.keys())
    all_pairs = [(src_uri, tgt_uri) for src_uri in source_uris for tgt_uri in target_uris]
    return all_pairs


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