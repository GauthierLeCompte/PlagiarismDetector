import csv
import math
from random import shuffle
# import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap
import seaborn as sns
import pandas as pd

BANDS = 10
SHINGLES = 3
SIGNATURE_LENGTH = 100
TRESHHOLD = 0.7

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


def run_jaccard(articles):
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
                    jaccard[temp_tuple1] = jaccard_similarity(articles[article_id_1], articles[article_id_2])

    return jaccard


def jaccard_similarity(doc1, doc2):
    """
    Calculates the jaccard similarity between 2 articles
    :param article1: article 1 we want to compare
    :param article2: article 2 we want to compare
    :return: jaccard similarity between article 1 and article 2
    """
    # List the unique words in a document
    words_doc1 = set(doc1.lower().split())
    words_doc2 = set(doc2.lower().split())

    # Find the intersection of words list of doc1 & doc2
    intersection = words_doc1.intersection(words_doc2)

    # Find the union of words list of doc1 & doc2
    union = words_doc1.union(words_doc2)

    # Calculate Jaccard similarity score
    # using length of intersection set divided by length of union set
    return float(len(intersection)) / len(union)

def shingle2(articles, k):
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
        words = text.lower().split()

        for i in range(len(words) - k):
            newword= ""
            for j in range(i,i+k):
                newword += words[j]
                newword += " "
            newword = newword[:-1]
            shingle_set.add(newword)
        shingled_articles[id] = shingle_set

    return shingled_articles

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

        for i in range(len(text) - k):
            shingle_set.add(text[i:i+k])
        shingled_articles[id] = shingle_set

    return shingled_articles


def unionize(articles):
    """
    Takes union of all the values in a dictionary
    :param articles: Dictionary of articles we want to take the union of
    :return: a set of all the shingles
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
        shuffled_vocab = list(range(1, len(vocabulary) + 1))
        shuffle(shuffled_vocab)
        hashes.append(shuffled_vocab)
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
    candidates = set()
    non_candidates = set()

    for article1 in subvectors:
        for article2 in subvectors:
            if article1 != article2:
                match = False
                temp_tuple1 = (article1, article2)
                temp_tuple2 = (article2, article1)
                for subvec1 in subvectors[article1]:
                    for subvec2 in subvectors[article2]:
                        if subvec1 == subvec2:
                            match += True
                            if temp_tuple1 in candidates or temp_tuple2 in candidates:
                                pass
                            else:
                                candidates.add(temp_tuple1)
                            break
                if not match:
                    if temp_tuple1 in non_candidates or temp_tuple2 in non_candidates:
                        pass
                    else:
                        non_candidates.add(temp_tuple1)

    return candidates, non_candidates

def export_results(candidate_pairs, jaccard, similarity):
    end_result = {}
    scores = []
    for pair in candidate_pairs:
        score = jaccard[pair]

        if score >= similarity:
            end_result[pair] = score
            scores.append(score)

    with open('../output/results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Document 1", "Document 2", "score"])

        for pair in end_result:
            writer.writerow([pair[0], pair[1], end_result[pair]])
    return scores

def calc_probability(similarity, rows, bands):
    return 1 - (1 - similarity ** rows) ** bands

def bar_plot(jaccard):
    # creating the dataset
    valuelist = []
    percentlist = [0,0,0,0,0,0,0,0,0,0]
    for key in jaccard.keys():
        value = jaccard[key]
        smallval = math.floor(value*10)
        if smallval ==10:
            smallval=9
        percentlist[smallval]+=1
        valuelist.append(value*100)

    bin_edges = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    fig, ax = plt.subplots(1, 1)

    ax.hist(valuelist,
             bins=bin_edges,
             density=False,
             histtype='bar',
             color='b',
             edgecolor='k',
             alpha=0.5)

    ax.set_xlabel("Similarity between chapters (in %)")
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.set_ylabel("Number of chapters")
    ax.set_yscale('log')
    ax.set_title("\n".join(wrap("Number of chapters in function of their similarity witch each other (using 1000 articles)", 60)))
    rects = ax.patches
    labels = percentlist
    print(percentlist, "===========================")

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 0.01, label, ha='center', va='bottom')

    plt.savefig("histScore_shinling2Copy.png")
    plt.show()
    return valuelist

def plot_candidate_probability(candidate_pairs, non_candidate_pairs):
    results = pd.DataFrame({
        'similarity': [],
        'probability': [],
        'rows, bands': []
    })

    fig, axs = plt.subplots()
    for similarity in np.arange(0.01, 1, 0.01):
        total = SIGNATURE_LENGTH
        for band in [100,50,25, 20, 15, 10, 5, 2, 1]:
            rows = int(total / band)
            probability = calc_probability(similarity, rows, band)
            results = results.append({
                'similarity': similarity,
                'probability': probability,
                'rows, bands': f"{rows},{band}"},
                ignore_index=True)

    plot = sns.lineplot(data=results, x='similarity', y='probability', hue='rows, bands', ax=axs)
    axs2 = axs.twinx()

    x_coord = []
    y_coord = []
    for pair1 in candidate_pairs:
        score = jaccard[pair1]
        x_coord.append(score)
        y_coord.append(1)

    for pair2 in non_candidate_pairs:
        score = jaccard[pair2]
        x_coord.append(score)
        y_coord.append(0)

    axs2.scatter(x_coord, y_coord, s=3 , c="black")
    axs2.set_ylabel("candidates")
    axs2.set_yticks(np.arange(0, 1.1, 1.0))
    axs.set_title("\n".join(wrap("", 60)))


    plt.show()

if __name__ == '__main__':
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    small = "../input/news_articles_small.csv"
    large = "../input/news_articles_large.csv"

    ### Parsing
    articles = parse_csv(small)
    print(f"Articles Parsed\n")

    jaccard = run_jaccard(articles)
    ### Shingles
    shingled_articles = shingle(articles, SHINGLES)
    print(f"Length Shingles {len(shingled_articles)}\n")

    ### Generate vocabulary
    vocabulary = unionize(shingled_articles)
    print(f"Vocab length {len(vocabulary)}\n")

    ### Hot encoding
    hot_encoded_articles = one_hot_encoding(vocabulary, shingled_articles)
    print(f"Length Hot Encoded Articles {len(hot_encoded_articles)}\n")

    ### Min Hash
    minhash_func = build_minhash_func(vocabulary, SIGNATURE_LENGTH)
    print(f"Minhash function calculated\n")

    signatures = {}
    for article_id in hot_encoded_articles:
        signatures[article_id] = create_hash(hot_encoded_articles[article_id], minhash_func, vocabulary)
    print(f"Signatures created\n")

    ### Create barplot
    valuelist = bar_plot(jaccard)
    print(f"plot created")

    ### Locality Sensetive Hashing
    subvectors = {}
    for signature_id in signatures:
        subvectors[signature_id] = create_subvectors(signatures[signature_id], BANDS)
    print(f"Subvectors created\n")

    candidate_pairs, non_candidate_pairs = find_candidate_pairs(subvectors)
    print(f"Lenght Candidate pairs: {len(candidate_pairs)}" )
    print(f"Lenght Non-Candidate pairs: {len(non_candidate_pairs)}")
    print(f"Lenght both: {len(candidate_pairs) + len(non_candidate_pairs)}")

    ### Export results
    scores = export_results(candidate_pairs, jaccard, TRESHHOLD)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
