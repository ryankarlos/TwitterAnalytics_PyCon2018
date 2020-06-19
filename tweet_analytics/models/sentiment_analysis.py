import pandas as pd

from tweet_analytics.models import (
    clustering,
    feature_matrix,
    model_build,
    model_predict_evaluate,
    model_train,
    name_DT,
    oversampling,
    param_grid_DT,
    plot_confusion_matrix,
    sentiment_labels,
    split_train_test,
    tsne_visualiser,
    tweet_sentiment_label,
)

# ---------------- Importing examples for clustering and t-sne ------------------


tweet_df = pd.read_csv("snakebite.csv", sep=",", encoding="latin-1")
tweet_df = tweet_sentiment_label(tweet_df, sentiment_labels)
x_array = tweet_df["cleaned"].to_numpy()
features = feature_matrix(x_array, (1, 3), "tfidf")
clustering(features["feat_names"], features["feat_matrix"])
tsne_visualiser(features["feat_matrix"], tweet_df["label"])


# ---------------- Feature Engineering/ Model Training and Evaluation  ------------------


train_test = split_train_test(tweet_df)
features = feature_matrix(train_test["x_train"], (0, 1), "tfidf")
x_train_vectors = features["feat_matrix"]
x_test_vectors = features["vectoriser"].transform(train_test["x_test"])
x_resampled, y_resampled = oversampling(x_train_vectors, train_test["y_train"])
clf_DT = model_build(name=name_DT)
clf_best = model_train(x_resampled, y_resampled, clf_DT, params=param_grid_DT)
predictions = model_predict_evaluate(
    x_resampled, x_test_vectors, y_resampled, train_test["y_test"], clf_best, name_DT
)
plot_confusion_matrix(
    train_test["y_test"],
    predictions["prediction_test"],
    sentiment_labels,
    title="Confusion Matrix for Decision Tree",
)
