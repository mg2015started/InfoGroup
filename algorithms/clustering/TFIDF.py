import string
import numpy as np
import pandas as pd
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
import matplotlib.colors as colors
from sklearn.metrics import calinski_harabaz_score
from scipy.cluster.hierarchy import ward, dendrogram
import matplotlib.pyplot as plt

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
    kl = np.sum(np.where(a * b != 0, a * np.log((a+1e-8) / (b+1e-8)), 0))
    return kl


def cosine_similarity(a, b):
    """
    cal cosine similarity between two vectors
    @param a: vector1
    @param b: vector2
    @return: cosine similarity
    """
    return 1-np.sum(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8))


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


def batch_distance(a, bs, dist_func):
    """
    cal distance between vector and vector-s
    @param a: vector1
    @param bs: vector2 s
    @return: distance vector
    """
    akls = [dist_func(a, b) for i, b in bs.iterrows()]
    return np.array(akls)


def use_DBSCAN(dist):
    """
    :param dist: data distance matrix
    :return: DBSCAN clustering algorithm
    This method is invalid, we find almost every datapoint is clustered to the same cluster.
    """
    clustering = DBSCAN(eps=0.0053, metric='precomputed', min_samples=3).fit(dist)
    labels = clustering.labels_
    print(labels)


def use_hierarchy(dist):
    """
    :param dist: data distance matrix
    :return: hierarchical clustering algorithm, and plot the cluster
    refer to Dai Xinyi's code
    """
    linkage_matrix = ward(dist)
    plt.figure(figsize=(15, 20))
    dendrogram(linkage_matrix, orientation='right', labels=dataset['title'].tolist())
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.tight_layout()
    # plt.show()
    plt.savefig('ward_cluster.png')


def kmeans(data, K, dist_func='cosine', delta=1e-8):
    """
    :param data: featue,[text_number, feature_dim]([N, dim])
    :param K: cluster number
    :param dist_func: choose the distance measurement
    :param delta: when to stop algorithm
    :return: cluster label
    This method is invalid, since Kmeans is suitable for euclidian distance. If we use \\
    AKL or cosine-sim,we need to derive the formula to calculate the cluster center. \\
    Thus, we use K-medoids instead.
    """
    dim = data.shape[1]
    center = data[:K]

    def _cosine_similarity_(A, B):
        """
        :param A: [N, dim]
        :param B: [K, dim]
        :return: [N, K], cosine similarity
        """
        f1 = np.matmul(A, B.T)
        f2 = np.linalg.norm(A, axis=1, keepdims=True) * np.linalg.norm(B, axis=1, keepdims=True).T
        return 1 - f1 / (f2+1e-8)

    def _euclid_dist_(A, B):
        """
        :param A: [N, dim]
        :param B: [K, dim]
        :return: [N, K], euclid distance
        """
        a, b = A.shape[0], A.shape[1]
        k = B.shape[0]
        tmpA = np.reshape(np.tile(A, [1, k]), [a, k, b])
        return np.linalg.norm(tmpA-B, axis=-1) / 2

    pre_loss = epochs = 0
    delta_loss = 10000
    while delta_loss > delta:
        if dist_func == 'cosine':
            dist_data_center = _cosine_similarity_(data, center)
        elif dist_func == 'akl':
            dist_data_center = [[average_kl_divergence(x, y) for y in center] for x in data]
            dist_data_center = np.array(dist_data_center).astype(np.float32)
        else:
            dist_data_center = _euclid_dist_(data, center)

        labels = np.argmin(dist_data_center, axis=1)
        for i in range(K):
            idx = np.where(labels == i)
            if len(idx[0]) == 0:
                center[i] = np.zeros(dim)
            else:
                center[i] = np.mean(data[idx], axis=0)

        # calculate kmeans loss
        loss = np.sum(np.min(dist_data_center, axis=1))
        delta_loss = abs(loss - pre_loss)
        pre_loss = loss

        # epochs += 1
        # print('epoch:%d, loss:%.4f' % (epochs, loss))

    return labels, loss


def kmedoids(data, K, dist_matrix, max_iter=100):
    """
    :param data: featue,[text_number, feature_dim]([N, dim])
    :param K: cluster number
    :param dist_matrix: distance matrix of data
    :param max_iter: when to stop algorithm
    :return: cluster label
    """
    def _cluster_and_loss_(N, center):
        """
        :param N: data point number
        :param center: cluster center index. (cluster must be in the origin data point)
        :return: cluster resultes (labels), loss_
        """
        dist_data_center = [[dist_matrix[i][j] for j in center] for i in range(N)]
        dist_data_center = np.array(dist_data_center).astype(np.float32)
        labels_ = np.argmin(dist_data_center, axis=1)
        loss_ = np.sum(np.min(dist_data_center, axis=1))
        return labels_, loss_

    nsamples, dim = data.shape[0], data.shape[1]
    center = np.arange(K).astype(np.int32)

    labels, loss = _cluster_and_loss_(nsamples, center)
    for epoch in range(max_iter):
        nochange = 0
        for i in range(nsamples):
            if i in center:
                continue

            old_mediods = center[labels[i]]
            center[labels[i]] = i
            new_labels, new_loss = _cluster_and_loss_(nsamples, center)
            if new_loss < loss:
                labels = new_labels
                loss = new_loss
            else:
                center[labels[i]] = old_mediods
                nochange += 1
                if nochange == (nsamples-K):
                    return labels, loss

        print('epoch:%d, loss:%.4f' % (epoch, loss))
    return labels, loss


def plot(data, labels):
    """
    :param data:
    :param labels: cluster labels
    :return: visualize the kmeans cluster
    """
    # color_list = list(colors.cnames.keys())
    color_list = ['red', 'blue', 'green', 'orange', 'black']
    for i in range(data.shape[0]):
        plt.scatter(data[i][0], data[i][1], color=color_list[labels[i]])
    plt.savefig('kmeans_cluster.png')


if __name__ == '__main__':
    dataset, features = load_dataset("AAAI-14+Accepted+Papers.csv")
    contents = dataset['title'] + " " + dataset['keywords'] + " " + dataset['abstract']

    # # process data
    # contents = contents.apply(lambda x: process_text(x))
    #
    # # vectors for each document, shape: (|D|, |words|)
    # tf_idf_vector = tf_idf(contents)
    #
    # # dist_matrix = tf_idf_vector.apply(lambda x: batch_distance(x, tf_idf_vector, average_kl_divergence), axis=1)
    # dist_matrix = tf_idf_vector.apply(lambda x: batch_distance(x, tf_idf_vector, cosine_similarity), axis=1)
    # print(dist_matrix)

    # # covert to numpy array
    # dist_matrix_ = []
    # for ele in dist_matrix.values:
    #     dist_matrix_.append(ele)
    # dist_matrix_ = np.array(dist_matrix_).astype(np.float32)
    # np.save('cosine_matrix.npy', dist_matrix_)

    # load distance matrix, and use hierarchical clustering, try DBSCAN, use Kmediods
    feature = np.load('tfidf_vector.npy')
    dist_matrix = np.load('akl_matrix.npy')
    # print(dist_matrix)
    # # use_DBSCAN(dist_matrix)
    # use_hierarchy(dist_matrix)
    # labels, _ = kmedoids(feature, K=5, dist_matrix=dist_matrix)
    # score = calinski_harabaz_score(feature, labels)
    K_list = [2, 5, 10, 15, 20, 30, 40, 50, 80, 100]
    score = []
    for k in K_list:
        labels, _ = kmedoids(feature, K=k, dist_matrix=dist_matrix)
        score_ = calinski_harabaz_score(feature, labels)
        score.append(score_)
        print(k, score_)
    plt.bar(K_list, score)
    plt.ylabel('CH-score')
    plt.xlabel('cluster number k')
    plt.show()
    print(score)

    # Kmeans
    # load data tfidf feature, use kmeans clustering
    # feature = np.load('tfidf_vector.npy')
    # data = TSNE(random_state=9).fit_transform(feature)
    # print(feature.shape)

    # labels, _ = kmeans(feature, K=5, dist_func='euclid')
    #
    # K_list = [2, 5, 10, 15, 20, 30, 40, 50]
    # score = []
    # for k in K_list:
    #     # labels = KMeans(n_clusters=k, random_state=9).fit_predict(feature)
    #
    #     labels, _ = kmeans(feature, k, dist_func='cosine')
    #     score_ = calinski_harabaz_score(feature, labels)
    #     score.append(score_)
    #     print(k, score_)
    # plt.bar(K_list, score)
    # plt.ylabel('CH-score')
    # plt.xlabel('cluster number k')
    # plt.show()