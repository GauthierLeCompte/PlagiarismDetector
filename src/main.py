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

def shingle(text, k):
    shingle_set = set()

    for i in range(len(text) - k+1):
        shingle_set.add(text[i:i+k])

    return shingle_set

def unionize(articles):
    vocab = set()

    for article_id in articles:
        vocab = vocab.union(articles[article_id])

    return list(vocab)

def one_hot_encoding(vocab, articles):
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
    small = "../input/news_articles_small.csv"
    large = "../input/news_articles_large.csv"

    #TODO: Er staan soms echt heel rare tekens in de tekst, dit kan mss problemen geven voor latere shit
    articles = parse_csv(small)

    shingled_articles = {}
    for id in articles:
        shingled_articles[id] = shingle(articles[id], 2)

    vocabulary = unionize(shingled_articles)

    hot_encoded_articles = one_hot_encoding(vocabulary, articles)


    print("hellow")