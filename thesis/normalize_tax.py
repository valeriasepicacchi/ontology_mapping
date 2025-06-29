from rdflib import Graph, Namespace
from normalization import Normalizer
# Define namespaces
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
SKOSXL = Namespace("http://www.w3.org/2008/05/skos-xl#")

# Load the RDF graph
g = Graph()
g.parse("emtree_release_202501.xml", format="xml")

# Extract concepts and their labels
concepts = []
for concept in g.subjects(predicate=SKOSXL.prefLabel):
    label_resource = g.value(subject=concept, predicate=SKOSXL.prefLabel)
    label = g.value(subject=label_resource, predicate=SKOSXL.literalForm)
    concepts.append({
        "URI": str(concept),
        "Preferred Label": str(label) if label else None
    })

# Create a pandas DataFrame
import pandas as pd
df_concepts = pd.DataFrame(concepts)
print(df_concepts.head())
# Instantiate the Normalizer
normalizer = Normalizer(replace_with_whitespace=True, end_only=False)

# Specify normalization operations (choose the ones relevant to your case)
operations = ["replaceaccents", "removequalifiers", "trimwhitespaces", "removepunctuation", "stem"]

# Normalize the labels
normalized_labels = normalizer.normalize(df_concepts["Preferred Label"].tolist(), operations)

# Add normalized labels to the DataFrame
df_concepts["Normalized Label"] = df_concepts["Preferred Label"].map(normalized_labels)
print(df_concepts.head())
# Example: Add dummy categories for training
df_concepts["Category"] = "chemical"  # Replace with actual categories based on your taxonomy
print(df_concepts.head())
# Prepare the training set
training_set = df_concepts[["Preferred Label", "Normalized Label", "Category"]]
print(training_set.head())
# Save to CSV
training_set.to_csv("taxonomy_training_set.csv", index=False)
print("Training set saved to taxonomy_training_set.csv")
