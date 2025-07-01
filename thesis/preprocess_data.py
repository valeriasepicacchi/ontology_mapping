from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS
import uuid
import pandas as pd
import random 

def load_xml(path):
    # Define namespaces
    SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
    SKOSXL = Namespace("http://www.w3.org/2008/05/skos-xl#")
    DCT = Namespace("http://purl.org/dc/terms/")
    # Load the RDF graph
    g = Graph()
    g.parse(path, format="xml")  # Adjust the file name
    # Extract concepts
    concepts = []
    for concept in g.subjects(predicate=SKOSXL.prefLabel):
        # Extract the Label element inside prefLabel
        label_resource = g.value(subject=concept, predicate=SKOSXL.prefLabel)
        # Extract the literal form of the label
        label = g.value(subject=label_resource, predicate=SKOSXL.literalForm)
        # Get the definition (if available)
        definition = g.value(subject=concept, predicate=SKOSXL.definition)
        # Get alternate labels
        alt_labels = list(g.objects(subject=concept, predicate=SKOSXL.altLabel))
        # Collect concept data
        concepts.append({
            "URI": str(concept),
            "Preferred Label": str(label) if label else "N/A",
            "Definition": str(definition) if definition else "N/A",  # Handle missing definition
            "Alternate Labels": [str(alt_label) for alt_label in alt_labels]
        })

    # Convert to DataFrame
    df = pd.DataFrame(concepts)
    return df

def create_subconcepts_exactmatching():

    # Define SKOS namespace
    SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
    SKOSXL = Namespace("http://www.w3.org/2008/05/skos-xl#")    
    # Load the original SKOS RDF graph
    g = Graph()
    g.parse("emtree_release_202501.xml", format="xml")  # Replace with your SKOS file path

    # Create a new graph to store the modified data
    new_graph = Graph()

    # Iterate over each concept in the original SKOS file
    c = 0
    for concept in g.subjects(predicate=SKOSXL.prefLabel):
        # Get the label of the concept (there could be multiple labels)
        label = g.value(subject=concept, predicate=SKOSXL.prefLabel)
        
        # If the concept has multiple labels (say, a comma-separated string), split the labels
        if label and "," in label:
            labels = label.split(",")  # Example: splitting by commas
            labels = [l.strip() for l in labels]  # Clean up extra spaces
        else:
            labels = [label]  # If only one label, we just keep it

        # Collect all labels (including alternate labels)
        for l in g.objects(subject=concept, predicate=SKOSXL.altLabel):
            labels.append(str(l))  # Ensure the alternate labels are added as strings
        
        # Ensure that each new sub-concept has at least 3 labels
        # if len(labels) < 3:
        #     # If fewer than 3 labels, pad with definitions or other relevant info
        #     definition = g.value(subject=concept, predicate=SKOSXL.definition)
        #     if definition:
        #         labels.append(str(definition))  # Add the definition as a label if available
        #     while len(labels) < 3:
        #         labels.append("Placeholder Label")  # Add placeholder labels to make sure we have at least 3
        
        # Create new concepts based on the labels
        new_concepts = []
        for i in range(2, len(labels),3):
            
            # Generate a new URI for each new concept
            new_concept_uri = URIRef(f"http://example.com/{uuid.uuid4()}")

            # Add the new concept to the new graph
            new_graph.add((new_concept_uri, RDF.type, SKOS.Concept))
            new_graph.add((new_concept_uri, SKOSXL.prefLabel, Literal(labels[i-2])))
            new_graph.add((new_concept_uri, SKOSXL.altLabel, Literal(labels[i-1])))
            new_graph.add((new_concept_uri, SKOSXL.altLabel, Literal(labels[i])))
            
            # Add each new concept to the list
            new_concepts.append(new_concept_uri)
            if i>=len(labels)-3:
                new_concept_uri = URIRef(f"http://example.com/{uuid.uuid4()}")

                # Add the new concept to the new graph
                new_graph.add((new_concept_uri, RDF.type, SKOS.Concept))
                
                new_graph.add((new_concept_uri, SKOSXL.prefLabel, Literal(labels[len(labels) -i-1])))
                for l in range(1,len(labels) -i):
                    new_graph.add((new_concept_uri, SKOSXL.altLabel, Literal(labels[len(labels) -i +l-1])))



        
        # Now, create relationships between the new concepts to link them back to the original concept
        # We can use `skosXL:exactMatch` or `skosXL:related` for the relationship.
        for i in range(len(new_concepts)):
            for j in range(i + 1, len(new_concepts)):
                # Add an exact match relationship between the two new concepts
                new_graph.add((new_concepts[i], SKOS.exactMatch, new_concepts[j]))
                new_graph.add((new_concepts[j], SKOS.exactMatch, new_concepts[i]))
        
        # Remove the original concept from the new graph (since we're replacing it)
        new_graph.remove((concept, None, None))

        if c == 10:
            continue
        else:
            c+= 1
            # print(c)

    # Save the new graph to a file
    new_graph.serialize(destination="new_taxonomy_skos.xml", format="xml")

    print("New SKOS file has been created: 'new_taxonomy_skos.xml'")

def create_nonmatching(num_pairs):
    SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
    SKOSXL = Namespace("http://www.w3.org/2008/05/skos-xl#")    

    # Load the original SKOS RDF graph
    g = Graph()
    g.parse("new_taxonomy_skos.xml", format="xml")  # Replace with your SKOS file path

    # Create a new graph to store the modified data
    new_graph = Graph()

    # Collect all concepts (this could be your unrelated or narrow relationship pool)
    concepts = list(g.subjects(predicate=SKOSXL.prefLabel))
    for _ in range(num_pairs):
        # Select two random concepts (ensuring they're not the same)
        concept1, concept2 = random.sample(concepts, 2)
        exact_match_exists = False
        for match in g.objects(subject=concept1, predicate=SKOS.exactMatch):
            if match == concept2:
                print('The concepts are the same')
                exact_match_exists = True
                break
        
        # If an exact match exists, skip creating a non-match
        if exact_match_exists:
            continue

        # Retrieve the labels for these concepts
        label1 = g.value(subject=concept1, predicate=SKOSXL.prefLabel)
        label2 = g.value(subject=concept2, predicate=SKOSXL.prefLabel)

        # Check if the labels exist (to avoid NoneType issues)
        label1 = str(label1) if label1 else "Unknown Label 1"
        label2 = str(label2) if label2 else "Unknown Label 2"

        # Create new relationships between these concepts
        new_graph.add((concept1, SKOSXL.narrowMatch, concept2))  # Link them as narrowMatch (non-match)
        new_graph.add((concept2, SKOSXL.narrowMatch, concept1))  # Link in reverse direction as well

        # Add alternate labels for non-matching concepts (optional)
        # new_graph.add((concept1, SKOSXL.altLabel, Literal(f"Non-match for {label2}")))
        # new_graph.add((concept2, SKOSXL.altLabel, Literal(f"Non-match for {label1}")))

        # Optionally, add other information or relationships if needed
        # For example, add `skosXL:related` for broader/narrower concept relationships
        # new_graph.add((concept1, SKOSXL.related, concept2))
        # new_graph.add((concept2, SKOSXL.related, concept1))

    # Serialize the new graph to a file
    new_graph.serialize(destination="new_taxonomy.xml", format="xml")
    print("New SKOS file with non-matches has been created: 'new_taxonomy.xml'")


# Uncomment to run the function
create_subconcepts_exactmatching()
create_nonmatching(5)
# Uncomment to load XML as a DataFrame
# df = load_xml("emtree_release_202501.xml")
# print(df)
