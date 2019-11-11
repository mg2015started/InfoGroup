import string

import numpy as np
import pandas as pd
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

"""
1. preprocess the text
2. transform text to feature vector
3. cal distance according to distance metrics
4. cluster according to some methods
"""


def load_dataset(file_name):
    """
    load csv dataset
    @param file_name: string
    @return: dataset: DataFrame
    """
    dataset = pd.read_csv(file_name, delimiter=',', encoding="ISO-8859-1")
    features = dataset.columns
    return dataset, features


def get_wordnet_pos(tag):
    """
    get word pos like v, noun, adv
    @param tag: string
    @return: wordnet tag
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def process_text(text):
    """
    preprocess text
    @param text:
    @return:
    """
    # to lower:
    text = text.lower()

    # remove punctuation
    remove = str.maketrans('', '', string.punctuation)
    text = text.translate(remove)

    # segment
    text = word_tokenize(text)

    # remove usual words
    text = [w for w in text if not w in stopwords.words("english")]

    # stemming or lemmatization
    # s = SnowballStemmer('english')
    # text = [s.stem(w) for w in text]
    tagged_sent = pos_tag(text)
    s = WordNetLemmatizer()
    text = [s.lemmatize(tagged[0], pos=get_wordnet_pos(tagged[1])) for tagged in tagged_sent]
    # s = WordNetLemmatizer()
    # text = [s.lemmatize(w) for w in text]

    return text


def tf_idf(contents):
    """
    cal tf idf matrix of documents
    @param contents: DataFrame
    @return: tfidf matrix: DataFrame (|D|, |Words|)
    """
    # sklearn way
    # corpus = ["This is sample document.", "another random document.", "third sample document text"]
    # vector = TfidfVectorizer()
    # tf_data = vector.fit_transform(corpus)
    # df1 = pd.DataFrame(tf_data.toarray(), columns=vector.get_feature_names())  # to DataFrame

    # handwriting
    """
    ----  word1  word2   word3   ...
    doc1  1      2       0
    doc2
    doc3
    """
    # get tf(d,t) matrix
    tf = contents.apply(lambda x: pd.Series(x).value_counts(normalize=True))
    tf = tf.fillna(0)

    # get df(t) matrix
    df = (tf > 0).sum(axis=0)

    # get |D|
    D = len(tf)

    # get tf-idf
    tfidf = tf * np.log(D / df)
    # print (tf, np.log(D/df), tfidf)
    return tfidf


def kl_divergence(a, b):
    """
    cal kl of two category distribution
    @param a: vector1
    @param b: vector2
    @return: kl
    """
    # scipy way
    # kl = scipy.stats.entropy(a, b)
    # todo deal with zero
    kl = np.sum(np.where(a * b != 0, a * np.log(a / b), 0))
    return kl


def average_kl_divergence(a, b):
    """
    cal avg kl of two distribution
    @param a: vector1
    @param b: vector2
    @return: avg kl
    """
    # todo deal with choosing lambda
    lamb = 0.5
    M = lamb * a + (1 - lamb) * b
    akl = lamb * kl_divergence(a, M) + (1 - lamb) * kl_divergence(b, M)
    return akl


def batch_average_kl_divergence(a, bs):
    """
    cal avg kl between distribution and distributions
    @param a: vector1
    @param bs: vector2 s
    @return: avg kls
    """
    akls = [average_kl_divergence(a, b) for i, b in bs.iterrows()]
    return np.array(akls)


if __name__ == '__main__':
    dataset, features = load_dataset("AAAI-14+Accepted+Papers.csv")
    contents = dataset['title'] + " " + dataset['keywords'] + " " + dataset['abstract']

    # process data
    contents = contents.apply(lambda x: process_text(x))

    # vectors for each document, shape: (|D|, |words|)
    tf_idf_vector = tf_idf(contents)
    kl_matrix = tf_idf_vector.apply(lambda x: batch_average_kl_divergence(x, tf_idf_vector), axis=1)
    print(kl_matrix)
