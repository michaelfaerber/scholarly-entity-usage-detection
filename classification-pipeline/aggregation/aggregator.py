from statistics import mean
from typing import List, Tuple, Dict

from nltk.corpus import stopwords
from nltk.corpus import wordnet


def aggregate_probabilities(probabilities: Dict[str, Tuple[List[float], str]]) -> List[Tuple[str, float, int, str]]:
    """
    For each entity, aggregates multiple probabilities to a single score.
    :param probabilities:
    :return: list with entities, a sentence for context, its probability and its count
    """
    return [(ner, mean(probabilities[ner][0]), len(probabilities[ner][0]), probabilities[ner][1]) for ner in probabilities]


def filter_entities(probabilities: List[Tuple[str, float, int, str]]) -> List[Tuple[str, float, int, str]]:
    """
    Filters the given probabilities list to only include used entities.
    :param probabilities:
    :return: list with used entities and its probability
    """
    #probabilities = [x for x in probabilities if x[1] > 0.5]
    probabilities = [x for x in probabilities if x[0].lower() not in stopwords.words('english')]
    probabilities = [x for x in probabilities if not wordnet.synsets(x[0])]
    return sorted(probabilities, key=lambda x: x[1], reverse=True)