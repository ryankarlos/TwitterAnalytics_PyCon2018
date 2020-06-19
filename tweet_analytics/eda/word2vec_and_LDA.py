import ast
import warnings
from collections import Counter
from pprint import pprint

import matplotlib.pyplot as plt
import pandas as pd
import plotly
import pyLDAvis.gensim
from plotly.graph_objs import Layout, Scatter
from sklearn.manifold import TSNE

import gensim
from gensim import models
from gensim.models import Word2Vec, ldamodel

warnings.filterwarnings("ignore", category=DeprecationWarning)


def word_count(tweet_df, top):

    word_list = []
    sentences = tweet_df["cleaned"].tolist()
    for i in range(tweet_df["cleaned"].shape[0]):
        sentence = sentences[i]
        sentence = ast.literal_eval(sentence)
        word_list = word_list + sentence
        count_all = Counter()
        count_all.update(word_list)
        words_common = count_all.most_common(top)

    return words_common


def gensim_dict(tweet_df, keywords_list):

    sentence_list = []
    for i in range(tweet_df["cleaned"].shape[0]):
        word_list = ast.literal_eval(tweet_df["cleaned"][i])
        resultwords = [word for word in word_list if word.lower() not in keywords_list]
        sentence_list.append(resultwords)
        dictionary = gensim.corpora.Dictionary(sentence_list)

    return {sentence_list: sentence_list, dictionary: dictionary}


def bag_of_words_corpus(sentence_list, dictionary, **kwargs):

    count = 0
    for k, v in dictionary.iteritems():
        print(k, v)
        count += 1
        if count > 10:
            break

    dictionary.filter_extremes(kwargs[0], kwargs[1], kwargs[2])
    bow_corpus = [dictionary.doc2bow(doc) for doc in sentence_list]

    return bow_corpus


def tfidf_corpus(bow_corpus):
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    for doc in corpus_tfidf:
        pprint(doc)
        break

    return corpus_tfidf


def lda_model(corpus_tfidf, dictionary, num_topics=10, num_words=10, vis="off"):

    lda_tfidf = ldamodel.LdaModel(corpus_tfidf, num_topics)
    lda_tfidf.save("lda_model")
    topics = lda_tfidf.show_topics(num_topics, num_words, log=False, formatted=False)
    words = dictionary.token2id
    topics_decoded = dict()

    for i in range(len(topics)):
        topic_no = "Topic " + str(i)
        topics_decoded[topic_no] = {}
        v = topics[i][1]
        for j in range(len(v)):
            word = list(words.keys())
            [(list(words.values())).index(int(v[j][0]))]
            topics_decoded[topic_no][word] = v[j][1]

    if vis == "on":
        pyLDAvis.enable_notebook()
        pyLDAvis.display(pyLDAvis.gensim.prepare(lda_tfidf, corpus_tfidf, dictionary))

    return topics_decoded


def word2vec_model(
    sentence_list,
    size=300,
    window=2,
    min_count=5,
    workers=1,
    sg=0,
    keyword="ntd",
    topn=20,
):

    model_word2vec = Word2Vec(sentence_list, size, window, min_count, workers, sg)
    model_word2vec.wv.most_similar(keyword, topn)
    vocab = list(model_word2vec.wv.vocab)
    X_transformed = model_word2vec[vocab]

    return X_transformed, vocab


def tsne_model_fit(
    X_transformed, vocab, perplexity, n_iter, learning_rate=200, n_components=2
):

    results = pd.DataFrame(columns=["Perplexity", "Iterations", "Divergence"])

    for i in range(len(perplexity)):
        for j in range(len(n_iter)):
            tsne = TSNE(
                n_components=n_components,
                random_state=0,
                perplexity=perplexity[i],
                learning_rate=learning_rate,
                n_iter=n_iter[j],
                verbose=2,
            )
            tsne.fit_transform(X_transformed)
            results = results.append(
                {
                    "Perplexity": perplexity[i],
                    "Iterations": n_iter[j],
                    "Divergence": tsne.kl_divergence_,
                },
                ignore_index=True,
            )
            print("")
            print("")
            print("")
            print(
                "The KL divergence for perplexity: %d and "
                "iterations: %d is %.4f"
                % (perplexity[i], n_iter[j], tsne.kl_divergence_)
            )
            print("")
            print("")
            print("")

    tsne = TSNE(
        n_components=n_components,
        perplexity=results[results["Divergence"] == min(results["Divergence"])][
            "Perplexity"
        ].values[0],
        learning_rate=learning_rate,
        n_iter=int(
            results[results["Divergence"] == min(results["Divergence"])][
                "Iterations"
            ].values[0]
        ),
        verbose=2,
        random_state=42,
    )

    X_tsne = tsne.fit_transform(X_transformed)
    df = pd.concat([pd.DataFrame(X_tsne), pd.Series(vocab)], axis=1)
    df.columns = ["x", "y", "word"]

    return df


def plot_tsne(x, y, word):
    plt.figure(figsize=(15, 10))
    plt.margins(0)
    plt.axis("on")
    plt.title("Visualising Word2Vec embeddings using t-sne")
    for i, txt in enumerate(word):
        plt.annotate(txt, (x.iloc[i], y.iloc[i]))
    plt.scatter(
        x, y, color="g", cmap="hsv", alpha=0.9, marker="x", s=0.6, lw=1, edgecolor=""
    )
    plt.show()


def plotly_tsne(df, title="Visualising Word2Vec Embeddings using t-SNE"):

    trace = Scatter(
        x=df["x"],
        y=df["y"],
        mode="text",
        text=df["word"],
        textposition="bottom",
        marker=dict(color="#FFBAD2", line=dict(width=1)),
    )

    data = [trace]
    plotly.offline.plot({"examples": data, "layout": Layout(title=title)})
