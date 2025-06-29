from deeponto.onto import Ontology, OntologyPruner
from deeponto.align.mapping import *
# Load the DOID ontology
doid = Ontology("doid.owl")

# Initialise the ontology pruner
pruner = OntologyPruner(doid)

# Specify the classes to be removed
to_be_removed_class_iris = [
    "http://purl.obolibrary.org/obo/DOID_0060158",
    "http://purl.obolibrary.org/obo/DOID_9969"
]

# Perform the pruning operation
pruner.prune(to_be_removed_class_iris)

# Save the pruned ontology locally
pruner.save_onto("doid.pruned.owl")  

from deeponto.onto import Ontology
from deeponto.align.mapping import NegativeCandidateMappingGenerator, ReferenceMapping
from deeponto.align.bertmap import BERTMapPipeline

# Load the NCIT and DOID ontologies
ncit = Ontology("ncit.owl")
doid = Ontology("doid.owl")

# Load the equivalence mappings
ncit2doid_equiv_mappings = ReferenceMapping.read_table_mappings("ncit2doid_equiv_mappings.tsv")  # The headings are ["SrcEntity", "TgtEntity", "Score"]

# Load default config in BERTMap
config = BERTMapPipeline.load_bertmap_config()

# Initialise the candidate mapping generator
cand_generator = NegativeCandidateMappingGenerator(
  ncit, doid, ncit2doid_equiv_mappings, 
  annotation_property_iris = config.annotation_property_iris,  # Used for idf sample
  tokenizer=Tokenizer.from_pretrained(config.bert.pretrained_path),  # Used for idf sample
  max_hops=5, # Used for neighbour sample
  for_subsumptions=False,  # Set to False because the input mappings in this example are equivalence mappings
)

# Sample candidate mappings for each reference equivalence mapping
results = []
for test_map in ncit2doid_equiv_mappings:
    valid_tgts, stats = neg_gen.mixed_sample(test_map, idf=50, neighbour=50)
    print(f"STATS for {test_map}:\n{stats}")
    results.append((test_map.head, test_map.tail, valid_tgts))
results = pd.DataFrame(results, columns=["SrcEntity", "TgtEntity", "TgtCandidates"])
results.to_csv(result_path, sep="\t", index=False)