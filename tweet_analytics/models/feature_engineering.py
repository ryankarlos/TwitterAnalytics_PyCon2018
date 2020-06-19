import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from yellowbrick.text import TSNEVisualizer


def tweet_sentiment_label(tweet_df, *args):

    tweet_df["label"][tweet_df.loc[:, "Sentiment"] == 1, :] = args[0]
    tweet_df["label"][tweet_df.loc[:, "Sentiment"] == 0, :] = args[1]
    if len(args) > 2:
        tweet_df["label"].fillna(value=args[2], inplace=True)

    return tweet_df


def feature_matrix(x_array, grams, vectorizer):

    if vectorizer == "BOW":
        feature_vect = CountVectorizer(ngram_range=grams)
    elif vectorizer == "tfidf":
        feature_vect = TfidfVectorizer(use_idf=True, ngram_range=grams)
    else:
        raise ValueError("parameter vectorizer can only be 'BOW' or 'tfidf'")

    feat_matrix = feature_vect.fit_transform(x_array)
    feat_names = feature_vect.get_feature_names()

    return {
        "feat_names": feat_names,
        "feat_matrix": feat_matrix,
        "vectoriser": feature_vect,
    }


def clustering(feat_names, feat_matrix, num_clusters=10):
    km = KMeans(n_clusters=num_clusters)
    km.fit(feat_matrix)
    clusters = km.labels_.tolist()
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    for i in range(num_clusters):
        print("")
        print("Cluster {} : Words :".format(i))
        print(" %s" % [feat_names[ind] for ind in order_centroids[i, :10]])

    return clusters


def tsne_visualiser(feat_matrix, labels, tsne_vis=TSNEVisualizer):
    tsne = tsne_vis()
    tsne.fit(feat_matrix, labels)
    tsne.poof()


if __name__ == "__main__":

    sentiment_labels = ["Positive", "Negative", "Neutral"]
    tweet_df = pd.read_csv("snakebite.csv", sep=",", encoding="latin-1")
    tweet_df = tweet_sentiment_label(tweet_df, sentiment_labels)

    x_array = tweet_df["cleaned"].to_numpy()
    features = feature_matrix(x_array, (1, 3), "tfidf")
    clustering(features["feat_names"], features["feat_matrix"])
    tsne_visualiser(features["feat_matrix"], tweet_df["label"])
