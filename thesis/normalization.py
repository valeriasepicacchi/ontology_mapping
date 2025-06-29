from collections import OrderedDict
from typing import List
from core.constants import ERROR_MESSAGE, AVAILABLE_NORMALIZATION_OPERATIONS, LABEL_TRANSFORMATIONS_GRAPH, \
    CANDIDATE_POOL_GRAPH
from apis.URI_minting import uri_minter
from core.db.sparql import SPARQLInterface
from pandas import Series
from unidecode import unidecode
import nltk
import pandas as pd
import re
import core.logger as logging


logger = logging.get_logger(__name__)


class Normalizer(SPARQLInterface):
    """Normalization service takes parameters for functions at creation"""

    def __init__(self, replace_with_whitespace, end_only):
        super().__init__()
        self.replace_with_whitespace = replace_with_whitespace
        self.end_only = end_only

    def normalize(self, terms, operations: List[str], with_uri=False):
        """
        This function will unpack the operations and perform them for the terms.
        :param with_uri: If normalization has to be done involving URIs, this should be True.
        :param terms: List of terms for which the operation(s) is to be performed.
        :param operations: List of operations to be performed on the terms.
        :return: An object with input terms and their normalized terms.
        """
        try:
            logger.info("Operations specified: {}".format(operations))
            if not all(isinstance(item, str) for item in operations):
                return {ERROR_MESSAGE: "All the operations must be of type string."}, 400

            operations = [o.lower() for o in operations]
            operations.sort(reverse=True)

            # Check if any unsupported/invalid operations are provided
            if len(set(operations) - set(AVAILABLE_NORMALIZATION_OPERATIONS)) > 0:
                return {ERROR_MESSAGE: "Unrecognized operation found. Please provide operations from {}".format(
                    AVAILABLE_NORMALIZATION_OPERATIONS)}, 400

            # For normalization_with_uri
            uris = []
            if with_uri:
                if not all(isinstance(item["term"], str) for item in terms):
                    return {ERROR_MESSAGE: "All the terms must be of type string."}, 400

                # Create a uri list to attach with transformation resuls at the end.
                # Create an ordered dictionary of the terms so that URI and results of transformation will be in sync.
                # The key is the original term (and will stay as-is) and the value is the transformed term
                uris = [item["uri"] for item in terms]

                ordered_terms = OrderedDict()
                for item in terms:
                    ordered_terms.update({item["term"]: item["term"]})

                terms = ordered_terms
            else:
                if not all(isinstance(item, str) for item in terms):
                    return {ERROR_MESSAGE: "All the terms must be of type string."}, 400

                # Create a dictionary of the terms. The key is the original term (and will stay as-is) and the value
                # is the transformed term
                terms = dict(zip(terms, terms))

            # Perform the operations in the user-defined sequence
            for o in operations:
                try:
                    if o == "greekchars":
                        terms = self.replace_greek_characters(terms)
                    if o == "typographicforms":
                        terms = self.replace_typographic_forms(terms)
                    if o == "trimwhitespaces":
                        terms = self.remove_whitespace(terms)
                    if o == "stem":
                        terms = self.stem(terms)
                    if o == "replaceaccents":
                        terms = self.replace_accented_characters(terms)
                    if o == "lemmatize":
                        # WordNetLemmatizer requires downloading some files.
                        nltk.download('wordnet')
                        nltk.download('omw-1.4')
                        terms = self.lemmatize(terms)
                    if o == "tokenize":
                        terms = self.tokenize(terms)
                    if o == "removequalifiers":
                        terms = self.remove_qualifiers(terms)
                    if o == "removepunctuation":
                        terms = self.remove_punctuation(terms)
                    if o == "naturalwordorder":
                        terms = self.natural_word_order(terms)
                except Exception as e:
                    logger.error("Exception while performing operation: {}".format(str(e)))
                    continue

            # For normalization_with_uri
            if with_uri:
                final_lst = []
                for idx, (key, val) in enumerate(terms.items()):
                    final_lst.append({
                        "uri": uris[idx],
                        "inputTerm": key,
                        "transformedTerm": val
                    })

                return final_lst
            else:
                return terms
        except Exception as e:
            logger.error("Exception occurred: {}".format(str(e)))
            raise e

    def store_normalization(self, data: List[dict], operations: List[str], editor: str):
        try:
            column_names = ['concept_uri', 'source_uri', 'source_term', 'transformed_term', 'language_tag']
            transformed = pd.DataFrame(data, columns=column_names)

            # Create a new column in dataframe to mark if the transformed term exists in Candidate Pool or not.
            transformed["exists"] = self.term_existence_check(transformed["transformed_term"],
                                                              transformed["language_tag"],
                                                              include_case=True)

            # Delete all labels which already exist
            transformed = transformed[~transformed["exists"]]
            if len(transformed) > 0:
                today = self.getRDFDate()

                # Define the transformation run
                run_uri = uri_minter.get_hex_uri(1, num_of_characters=40, uri_base="candi:LabelTransformationRun-")[0]

                event_payload = f"""
                    {run_uri} a prov:Collection ;
                        prov:createdWith "{', '.join(operations)}" ;
                        pav:createdBy {editor} ;
                        pav:createdOn {today} ; \n
                """

                # If it doesn't exist, create a label ID for it, and add it to the proto-concept as an altLabel
                transformed["instance"] = uri_minter.get_hex_uri(len(transformed),
                                                                 num_of_characters=40,
                                                                 uri_base="candi:LabelTransformationInstance-")

                # Also create label URIs for new labels
                transformed["new_label_URI"] = uri_minter.get_hex_uri(len(transformed),
                                                                      num_of_characters=30,
                                                                      uri_base="elsvoc:Label-")

                instance_payload = transformed["instance"] + " prov:used " + transformed["source_uri"] + " ; "

                event_payload += " prov:hadMember " + ', '.join(transformed["instance"].tolist()) + " . "

                instance_payload += ' termsch:inputTerm "' + transformed["source_term"] + '" ; '
                instance_payload += ' termsch:outputTerm "' + transformed["transformed_term"] + '" . '
                instance_payload += transformed["new_label_URI"] + " prov:wasDerivedFrom " + transformed["instance"] + " . "

                instance_payload = ' '.join(instance_payload)

                # Create the new label itself, and link it to the proto-concept
                label_payload = transformed["new_label_URI"] + ' skosxl:literalForm "' + transformed["transformed_term"] + '"@' + \
                                                               transformed["language_tag"] + '; '
                label_payload += ' a skosxl:Label ; '
                label_payload += " pav:createdOn " + today + " ; "
                label_payload += " pav:lastUpdatedOn " + today + " ; "
                label_payload += ' pav:createdBy ' + editor + ' ; '
                label_payload += ' pav:curatedBy ' + editor + ' . '
                label_payload += transformed["concept_uri"] + " skosxl:altLabel " + transformed["new_label_URI"] + " . "

                label_payload = ' '.join(label_payload.tolist())

                payload = event_payload + instance_payload

                out, status = self.decorate_and_insert(payload={LABEL_TRANSFORMATIONS_GRAPH: payload,
                                                                CANDIDATE_POOL_GRAPH: label_payload})

                if not status:
                    return out, 500
                return out, 200
            else:
                out = {"message": "No new terms to be transformed."}
                return out, 200
        except Exception as e:
            logger.error("Exception: {}".format(str(e)))
            raise e

    def replace_greek_characters(self, terms):
        """Replaces greek characters with their name in Latin script using the list `greek characters`."""
        try:
            translations = {"\u0391": "Alpha",
                            "\u0392": "Beta",
                            "\u0393": "Gamma",
                            "\u0394": "Delta",
                            "\u0395": "Epsilon",
                            "\u0396": "Zeta",
                            "\u0397": "Eta",
                            "\u0398": "Theta",
                            "\u0399": "Iota",
                            "\u039a": "Kappa",
                            "\u039b": "Lambda",
                            "\u039c": "Mu",
                            "\u039d": "Nu",
                            "\u039e": "Xi",
                            "\u039f": "Omicron",
                            "\u03a0": "Pi",
                            "\u03a1": "Rho",
                            "\u03a3": "Sigma",
                            "\u03a4": "Tau",
                            "\u03a5": "Ypsilon",
                            "\u03a6": "Phi",
                            "\u03a7": "Chi",
                            "\u03a8": "Psi",
                            "\u03a9": "Omega",
                            "\u03b1": "alpha",
                            "\u03b2": "beta",
                            "\u03b3": "gamma",
                            "\u03b4": "delta",
                            "\u03b5": "epsilon",
                            "\u03b6": "zeta",
                            "\u03b7": "eta",
                            "\u03b8": "theta",
                            "\u03b9": "iota",
                            "\u03ba": "kappa",
                            "\u03bb": "lambda",
                            "\u03bc": "mu",
                            "\u03bd": "nu",
                            "\u03be": "xi",
                            "\u03bf": "omicron",
                            "\u03c0": "pi",
                            "\u03c1": "rho",
                            "\u03c2": "sigma",
                            "\u03c3": "sigma",
                            "\u03c4": "tau",
                            "\u03c5": "ypsilon",
                            "\u03c6": "phi",
                            "\u03c7": "chi",
                            "\u03c8": "psi",
                            "\u03c9": "omega"}
            out = OrderedDict(zip(list(terms.keys()), list(Series(terms.values()).replace(translations, regex=True))))
            return out
        except Exception as e:
            logger.error("Exception: {}".format(str(e)))
            return terms

    def replace_typographic_forms(self, terms):
        """Replaces curly single and double quotes with straight ones, and different dashes and minus with a simple \
        hyphen."""
        try:
            typographics = {'−': '-', '–': '-', '—': '-', '“': '"', '”': '"', '‘': "'", '’': "'"}
            typographics = str.maketrans(typographics)
            out = OrderedDict(zip(list(terms.keys()), [x.translate(typographics) for x in list(terms.values())]))
            return out
        except Exception as e:
            logger.error("Exception: {}".format(str(e)))
            return terms

    def remove_qualifiers(self, terms):
        """Removes 'qualifiers' that are enclosed in parentheses, e.g. (biology), from the string including adjacent \
        extra whitespace."""
        try:
            if self.end_only:
                pattern = r'\s*?\(.*?\)\s*?$'
            else:
                pattern = r'\s*?\(.*?\)\s*?'
            out = OrderedDict(zip(list(terms.keys()), [re.sub(pattern, "", x) for x in list(terms.values())]))
            return out
        except Exception as e:
            logger.error("Exception: {}".format(str(e)))
            return terms

    def remove_whitespace(self, terms):
        """Replaces any whitespace in the input string with a single space."""
        try:
            results = Series(terms.values()).str.strip()
            results = results.replace(r'\s{2,}', value=' ', regex=True)
            out = OrderedDict(zip(list(terms.keys()), results))
            return out
        except Exception as e:
            logger.error("Exception: {}".format(str(e)))
            return terms

    def remove_punctuation(self, terms):
        """Removes any punctuation from the string, including some non-ASCII."""
        try:
            pattern = r'[_\.,;:!?\*‐\-–—#<>\(\)\[\]„“”‘’\'\"/\\\|%\^~`$=\+{}\*\@\&′″、]'
            replacement = ' ' if self.replace_with_whitespace else ''
            out = OrderedDict(zip(list(terms.keys()), [re.sub(pattern, replacement, x) for x in terms.values()]))
            return out
        except Exception as e:
            logger.error("Exception: {}".format(str(e)))
            return terms

    def tokenize(self, terms):
        """Returns a list of tokens for the string. Uses by default the NLTK TreebankWordTokenizer but another one can
        be passed into the method through the `tokenize_function` parameter."""
        try:
            results = [nltk.TreebankWordTokenizer().tokenize(x) for x in list(terms.values())]
            out = OrderedDict(zip(list(terms.keys()), results))
            return out
        except Exception as e:
            logger.error("Exception: {}".format(str(e)))
            return terms

    def stem(self, terms):
        """Returns the stem of each word in the string.
        Uses by default the NLTK SnowballStemmer but another one can be passed
        into the method through the `stem_function` parameter.
        """
        try:
            results = [nltk.SnowballStemmer("english").stem(x) for x in list(terms.values())]
            out = OrderedDict(zip(list(terms.keys()), results))
            return out
        except Exception as e:
            logger.error("Exception: {}".format(str(e)))
            return terms

    def lemmatize(self, terms):
        """Returns the lemma of each word in the string.
        Uses by default the NLTK WordNetLemmatizer but another one can be passed
        into the method through the `lemmatize_function` parameter.
        """
        try:
            results = [nltk.WordNetLemmatizer().lemmatize(x) for x in list(terms.values())]
            out = OrderedDict(zip(list(terms.keys()), results))
            return out
        except Exception as e:
            logger.error("Exception: {}".format(str(e)))
            return terms

    def natural_word_order(self, terms):
        """Returns the string split on commas and reverses the order of tokens."""
        try:
            def splitAndReverse(term):
                splitted = re.split(r'\s?,\s?', term)
                if " " in splitted:
                    splitted.remove(" ")
                splitted.reverse()
                final = (' ').join(splitted)
                return final

            out = OrderedDict(zip(terms.keys(), [splitAndReverse(x) for x in terms.values()]))
            return out
        except Exception as e:
            logger.error("Exception: {}".format(str(e)))
            return terms

    def replace_accented_characters(self, terms):
        """Replaces accented, non-ASCII characters with an ASCII equivalent as defined in the list `accented
         characters`."""
        try:
            results = [unidecode(x) for x in list(terms.values())]
            out = OrderedDict(zip(list(terms.keys()), results))
            return out
        except Exception as e:
            logger.error("Exception: {}".format(str(e)))
            return terms
