import configparser
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from tweet_analytics.models import feature_matrix

Config = configparser.ConfigParser()
Config.read("config.ini")

sentiment_labels = Config.get("model_params", "sentiment_labels")
param_grid_SVM = Config.get("model_params", "param_grid_SVM")
param_grid_NB = Config.get("model_params", "param_grid_NB")
param_grid_DT = Config.get("model_params", "param_grid_DT")
name_DT = Config.get("model_params", "name_DT")
name_NB = Config.get("model_params", "name_NB")
name_SVM = Config.get("model_params", "name_SVM")


def split_train_test(tweet_df, test_size=0.25):

    tweet_df["Sentiment"].replace(0.5, 2, inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(
        tweet_df["cleaned"][tweet_df["Sentiment"].notnull()],
        tweet_df["Sentiment"][tweet_df["Sentiment"].notnull()],
        test_size=test_size,
        random_state=42,
    )

    return {"X_train": x_train, "X_test": x_test, "Y_train": y_train, "Y_test": y_test}


def oversampling(x_train_vec, y_train):

    x_resampled, y_resampled = SMOTE(random_state=42).fit_sample(x_train_vec, y_train)

    return x_resampled, y_resampled


def model_build(name):

    if name == "SVM":
        clf = SVC()
    elif name == "Decision Tree":
        clf = DecisionTreeClassifier()
    elif name == "Naive Bayes":
        clf = MultinomialNB(fit_prior=True)
    else:
        raise ValueError("parameter name not recognised")

    return clf


def model_train(train_vectors, y_train, clf, cv=5, **params):

    grid_search = GridSearchCV(clf, params, cv=cv)
    grid_search.fit(train_vectors, y_train)
    clf_best = grid_search.best_estimator_
    print(
        "SVM RBF: The highest accuracy on the training set after grid search was %.2f"
        % (grid_search.best_score_)
    )
    print("The best parameters are:")
    print(grid_search.best_params_)

    return clf_best


def model_predict_evaluate(
    train_vectors, test_vectors, y_train, y_test, clf_best, name
):

    prediction_train = clf_best.predict(train_vectors)
    prediction_test = clf_best.predict(test_vectors)
    print("Results for %s om training set:" % name)
    print("")
    print(classification_report(y_train, prediction_train))
    print("")
    print("Results for %s om test set:" % name)
    print("")
    print(classification_report(y_test, prediction_test))

    return {"prediction_train": prediction_train, "prediction_test": prediction_test}


def plot_confusion_matrix(y_test, prediction_test, classes, title, cmap=plt.cm.Blues):

    cnf_matrix = confusion_matrix(y_test, prediction_test)
    plt.imshow(cnf_matrix, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.figure(figsize=(7, 7))
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title(title)
    plt.show()


if __name__ == "__main__":

    tweet_df = pd.read_csv("snakebite.csv", sep=",", encoding="latin-1")
    train_test = split_train_test(tweet_df)
    features = feature_matrix(train_test["x_train"], (0, 1), "tfidf")
    x_train_vectors = features["feat_matrix"]
    x_test_vectors = features["vectoriser"].transform(train_test["x_test"])
    x_resampled, y_resampled = oversampling(x_train_vectors, train_test["y_train"])
    clf_DT = model_build(name=name_DT)
    clf_best = model_train(x_resampled, y_resampled, clf_DT, params=param_grid_DT)
    predictions = model_predict_evaluate(
        x_resampled,
        x_test_vectors,
        y_resampled,
        train_test["y_test"],
        clf_best,
        name_DT,
    )
    plot_confusion_matrix(
        train_test["y_test"],
        predictions["prediction_test"],
        sentiment_labels,
        title="Confusion Matrix for Decision Tree",
    )
