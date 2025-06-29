from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF
import uuid
import random

# Define Namespaces
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
SKOSXL = Namespace("http://www.w3.org/2008/05/skos-xl#")
DCT = Namespace("http://purl.org/dc/terms/")

def create_taxonomy_with_relationships(num_pairs):
    """ 
    Create new concepts with nested skosxl:Label elements and add broader, narrower, 
    exact matches (once every three concepts), and non-exact matches.
    """
    # Load the original SKOS RDF graph
    g = Graph()
    g.parse("emtree_release_202501.xml", format="xml")  # Replace with your actual file

    # Create a new graph for modified data
    new_graph = Graph()
    new_graph.bind("skos", SKOS)
    new_graph.bind("skosxl", SKOSXL)
    new_graph.bind("dct", DCT)

    # Store broader and narrower relationships
    broader_narrower_relationships = []
    new_concepts = []

    # Iterate over each concept in the SKOS file
    concepts = list(g.subjects(predicate=RDF.type, object=SKOS.Concept))
    for index, concept in enumerate(concepts):
        # Create a new concept URI
        new_concept_uri = URIRef(f"http://example.com/concept-{uuid.uuid4()}")

        # Add the new concept to the graph
        new_graph.add((new_concept_uri, RDF.type, SKOS.Concept))

        # Generate a new skosxl:Label for the preferred label
        label_uri = URIRef(f"http://example.com/label-{uuid.uuid4()}")
        new_graph.add((new_concept_uri, SKOSXL.prefLabel, label_uri))
        new_graph.add((label_uri, SKOSXL.literalForm, Literal(f"Preferred Label {index+1}", lang="en")))
        new_graph.add((label_uri, DCT.dateAccepted, Literal("2025-01-01")))
        new_graph.add((label_uri, SKOS.note, Literal(f"Note for Preferred Label {index+1}", lang="en")))

        # Generate alternate labels
        for i in range(2):  # Create 2 alternate labels for each concept
            alt_label_uri = URIRef(f"http://example.com/alt-label-{uuid.uuid4()}")
            new_graph.add((new_concept_uri, SKOSXL.altLabel, alt_label_uri))
            new_graph.add((alt_label_uri, SKOSXL.literalForm, Literal(f"Alternate Label {index+1}-{i+1}", lang="en")))
            new_graph.add((alt_label_uri, DCT.dateAccepted, Literal("2025-01-01")))
            new_graph.add((alt_label_uri, SKOS.note, Literal(f"Note for Alternate Label {index+1}-{i+1}", lang="en")))

        new_concepts.append(new_concept_uri)

        # Store broader/narrower relationships (if they exist in the original graph)
        broader = list(g.objects(subject=concept, predicate=SKOS.broader))
        narrower = list(g.objects(subject=concept, predicate=SKOS.narrower))
        broader_narrower_relationships.append((new_concept_uri, broader, narrower))

        # Create exact matches once every three concepts
        if len(new_concepts) >= 3 and len(new_concepts) % 3 == 0:
            new_graph.add((new_concepts[-3], SKOS.exactMatch, new_concepts[-2]))
            new_graph.add((new_concepts[-2], SKOS.exactMatch, new_concepts[-1]))
            new_graph.add((new_concepts[-1], SKOS.exactMatch, new_concepts[-3]))

        if index >= 20:  # Limit to 20 concepts for easier testing
            continue

    # Restore Broader/Narrower Relationships
    for concept, broader, narrower in broader_narrower_relationships:
        for b in broader:
            new_graph.add((concept, SKOS.broader, b))
        for n in narrower:
            new_graph.add((concept, SKOS.narrower, n))

    # Create Non-Exact Matches
    for _ in range(num_pairs):
        # Select two random concepts ensuring they are different
        concept1, concept2 = random.sample(new_concepts, 2)

        # Ensure they are not already exact matches
        exact_match_exists = any(match == concept2 for match in new_graph.objects(subject=concept1, predicate=SKOS.exactMatch))
        if exact_match_exists:
            continue

        # Add a non-exact match (narrowMatch)
        new_graph.add((concept1, SKOSXL.narrowMatch, concept2))
        new_graph.add((concept2, SKOSXL.narrowMatch, concept1))

    # Serialize the modified graph to an RDF/XML file
    new_graph.serialize(destination="new_taxonomy_skos.xml", format="xml")
    print("New SKOS file with nested skosxl:Label elements, exact, non-exact matches, and broader/narrower relationships created: 'new_taxonomy_skos.xml'")

# Run the function
create_taxonomy_with_relationships(num_pairs=100)
