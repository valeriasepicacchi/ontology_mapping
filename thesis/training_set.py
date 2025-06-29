from rdflib import Graph, Namespace
import pandas as pd

# Define namespaces
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
SKOSXL = Namespace("http://www.w3.org/2008/05/skos-xl#")
RDF = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")

# Load the RDF graph
g = Graph()
g.parse("emtree_release_202501.xml", format="xml")  # Adjust your file path

# Extract concepts
concepts = []
for concept in g.subjects(predicate=SKOSXL.prefLabel):
    # Preferred Label
    label_resource = g.value(subject=concept, predicate=SKOSXL.prefLabel)
    preferred_label = g.value(subject=label_resource, predicate=SKOSXL.literalForm)
    
    # Type (if available)
    concept_type = g.value(subject=concept, predicate=RDF.type)
    
    # Add to the list
    concepts.append({
        "URI": str(concept),
        "Preferred Label": str(preferred_label) if preferred_label else None,
        "Type": str(concept_type) if concept_type else None
    })

# Convert to pandas DataFrame
df_concepts = pd.DataFrame(concepts)
print(df_concepts.head())
# Example: Group concepts into categories based on their type
df_concepts["Category"] = df_concepts["Type"].apply(lambda x: str(x).split("/")[-1] if pd.notnull(x) else None)#.apply(lambda x: "Chemical" if "chemical" in str(x).lower() else "Other")
for concept in g.subjects(predicate=SKOSXL.prefLabel):
    # Alternate labels
    alt_labels = list(g.objects(subject=concept, predicate=SKOS.altLabel))
    alt_labels = [str(label) for label in alt_labels]
    
    # Add to the DataFrame
    df_concepts.loc[df_concepts["URI"] == str(concept), "Alternate Labels"] = ", ".join(alt_labels)
training_set = df_concepts[["Preferred Label", "Category"]]

# If alternate labels are present, include them
if "Alternate Labels" in df_concepts.columns:
    training_set["Text"] = df_concepts["Preferred Label"] + " " + df_concepts["Alternate Labels"]
else:
    training_set["Text"] = df_concepts["Preferred Label"]

print(training_set.head())
training_set.to_csv("taxonomy_training_set.csv", index=False)
print("Training set saved to taxonomy_training_set.csv")
