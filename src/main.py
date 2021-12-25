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



if __name__ == '__main__':
    small = "../input/news_articles_small.csv"
    large = "../input/news_articles_large.csv"

    articles = parse_csv(small)

    shingled_articles = {}
    for id in articles:
        shingled_articles[id] = (shingle(articles[id], 2))

    print("hello there")