import csv
from random import shuffle
import matplotlib.pyplot as plt

def parse_csv(article):
    """
    Parse the given CSV file
    :param article: The small or large CSV file given as parameter
    :return: dictionary of articles where the key is the article number and the value the text of the article
    """
    articles = {}
    with open(article) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                articles[row[0]] = row[1]
            line_count += 1
    return articles

def run_jaccard():
    """
    Run jaccard similarity for all articles and call the function
    :return: dictionary where key's are the tuples and value the jaccard similarity
    """
    jaccard = {}
    for article_id_1 in articles:
        for article_id_2 in articles:
            if article_id_1 != article_id_2:
                temp_tuple1 = (article_id_1, article_id_2)
                temp_tuple2 = (article_id_2, article_id_1)

                if temp_tuple1 in jaccard or temp_tuple2 in jaccard:
                    pass
                else:
                    jaccard[temp_tuple1] = jaccard_similarity(set(articles[article_id_1]), set(articles[article_id_2]))
    return jaccard

def jaccard_similarity(article1, article2):
    """
    Calculates the jaccard similarity between 2 articles
    :param article1: article 1 we want to compare
    :param article2: article 1 we want to compare
    :return: jaccard similarity between article 1 and article 2
    """
    return float(len(article1.intersection(article2)) / len(article1.union(article2)))

def shingle(articles, k):
    """
    Shingling splits the text up into tokens of size k, with no duplicates
    :param articles: The articles we want to split up
    :param k: the length of the tokens
    :return: Set of all the shingles
    """
    shingled_articles = {}
    for id in articles:
        shingle_set = set()
        text = articles[id]

        for i in range(len(text) - k): # K+1 ofni?
            shingle_set.add(text[i:i+k])
        shingled_articles[id] = shingle_set

    return shingled_articles

def unionize(articles):
    """
    Takes union of all the values in a dictionary
    :param articles: Dictionary of articles we want to take the union of
    :return: a list of all the values
    """
    vocab = set()

    for article_id in articles:
        vocab = vocab.union(articles[article_id])

    return list(vocab)

def one_hot_encoding(vocab, articles):
    """
    This function will loop over all the articles and items in the vocab list, and adds a 1 if the element exists in
    the text of a article, otherwise it will add a 0. At the end of this function every article will have a corresponding
    list with 0's or 1's which indicates whether the corresponding vocabulary item exists in the text or not.
    :param vocab: The vocabulary which is a union of all the shingles of all the articles
    :param articles: All the articles we want to check
    :return: Dictionary where the key is the article id and the values are the hot encodings
    """
    hot_encoded_articles = {}
    for article_id in articles:
        temp_hot_encoded = []
        for x in vocab:
            if x in articles[article_id]:
                temp_hot_encoded.append(1)
            else:
                temp_hot_encoded.append(0)

        hot_encoded_articles[article_id] = temp_hot_encoded

    return hot_encoded_articles

def build_minhash_func(vocabulary, amount_hashes):
    """
    Builds vectors we use to compute the minhash function. Takes the vocabulary and shuffles it.
    The larger amount_hashes, the more accurate this will be
    :param vocabulary: The vocabulary we want to shuffle
    :param amount_hashes: The amount of hash vectors we want to create
    :return: list of hash vectors
    """
    # function for building multiple minhash vectors
    hashes = []
    for i in range(amount_hashes):
        hash_ex = list(range(1, len(vocabulary) + 1))
        shuffle(hash_ex)
        hashes.append(hash_ex)
    return hashes

def create_hash(hot_encoded_article, minhash_func, vocab):
    """
    Creates the signature or an hot encoded article (does the matching process)
    :param hot_encoded_article: The hot encoded article we want to create signature of
    :param minhash_func: List of hash vectors
    :param vocab: the vocabulary
    :return:
    """
    signature = []
    for func in minhash_func:
        for i in range(1, len(vocab)+1):
            index = func.index(i)
            signature_val = hot_encoded_article[index]
            if signature_val == 1:
                signature.append(i)
                break
    return signature

def create_subvectors(signature, band):
    """
    Creates subvectors of the signature vector
    :param signature: The signature vector we want to split
    :param band: Amount of times we want to split up the vector
    :return: The subvectors
    """
    length = len(signature)
    subvectors = [signature[i * length // band: (i + 1) * length // band] for i in range(band)]

    return subvectors

def find_candidate_pairs(subvectors):
    """
    Returns all candidate pairs (subvectors with the same value)
    :param subvectors: list of all the subvectors
    :return: all candidate pairs
    """
    #TODO: Hmm not sure geeft heel veel resultaten terug
    duplicates = set()

    for article1 in subvectors:
        for article2 in subvectors:
            if article1 != article2:
                for subvec1, subvec2 in zip(subvectors[article1], subvectors[article2]):
                    if subvec1 == subvec2:
                        temp_tuple1 = (article1, article2)
                        temp_tuple2 = (article2, article1)
                        if temp_tuple1 not in duplicates and temp_tuple2 not in duplicates:
                            duplicates.add(temp_tuple1)
                            print(f"Candidate pair: {subvec1} == {subvec2}")
                        else:
                            print(f"{temp_tuple1} zit er al in!!!!")
                        # we only need one band to match
                        break

    return duplicates

if __name__ == '__main__':
    small = "../input/news_articles_small.csv"
    large = "../input/news_articles_large.csv"

    ### Parsing
    articles = parse_csv(small)

    ### Jaccard Similarity
    jaccard = run_jaccard()

    ### Shingles
    shingled_articles = shingle(articles, 2)

    ### Generate vocabulary
    vocabulary = unionize(shingled_articles)

    ### Hot encoding
    hot_encoded_articles = one_hot_encoding(vocabulary, articles)

    ### Min Hash
    minhash_func = build_minhash_func(vocabulary, 100)

    signatures = {}
    for article_id in hot_encoded_articles:
        signatures[article_id] = create_hash(hot_encoded_articles[article_id], minhash_func, vocabulary)

    # Jaccard vergelijkingstest minhash
    '''jaccard2 = []
    for article_id_1 in signatures:
        for article_id_2 in signatures:
            jaccard2.append(jaccard_similarity(set(signatures[article_id_1]), set(signatures[article_id_2])))

    for i in range(10):
        print(f"Jaccard 1: {jaccard[i]} vs Jaccard 2: {jaccard2[i]}")'''

    # Locality Sensetive Hashing
    subvectors = {}
    for signature_id in signatures:
        subvectors[signature_id] = create_subvectors(signatures[signature_id], 10)

    candidate_pairs = find_candidate_pairs(subvectors)

    x=9
    print(x)
