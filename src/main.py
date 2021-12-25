import csv

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

def jaccard_similarity(article1, article2):
    """
    Calculates the jaccard similarity between 2 articles
    :param article1: article 1 we want to compare
    :param article2: article 1 we want to compare
    :return: jaccard similarity between article 1 and article 2
    """
    set1 = set(article1)
    set2 = set(article2)
    return float(len(set1.intersection(set2)) / len(set1.union(set2)))

def shingle(text, k):
    """
    Shingling splits the text up into tokens of size k, with no duplicates
    :param text: The text we want to split up
    :param k: the length of the tokens
    :return: Set of all the shingles
    """
    shingle_set = set()

    for i in range(len(text) - k+1):
        shingle_set.add(text[i:i+k])

    return shingle_set

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

if __name__ == '__main__':
    ### Parsing
    small = "../input/news_articles_small.csv"
    large = "../input/news_articles_large.csv"

    articles = parse_csv(small)

    ### Calculate Jaccard Similarity from the articles
    jaccard = []
    for article_id_1 in articles:
        for article_id_2 in articles:
            jaccard.append(jaccard_similarity(articles[article_id_1], articles[article_id_2]))


    ### Calculate Shingles from articles
    shingled_articles = {}
    for id in articles:
        shingled_articles[id] = shingle(articles[id], 2)

    ### Generate vocabulary from shingled articles
    vocabulary = unionize(shingled_articles)

    ### Hot encoding of the articles
    hot_encoded_articles = one_hot_encoding(vocabulary, articles)

    print("hellow")