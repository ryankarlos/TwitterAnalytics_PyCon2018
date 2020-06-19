import logging
import re
import string

import nltk
import pandas as pd
from dateutil.parser import parse
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from TwitterSearch import TwitterSearch, TwitterSearchException, TwitterSearchOrder

nltk.download("stopwords")
logger = logging.getLogger(__name__)


class TweetClean(object):
    def __init__(
        self,
        tweets_df,
        exclude_users=True,
        exclude_emojis=True,
        tokenise=True,
        lemmatise=True,
        custom_stopwords=None,
        split_timestamp=True,
    ):
        self.tweet_df = tweets_df
        self.exclude_emojis = exclude_emojis
        self.exclude_users = exclude_users
        if custom_stopwords is None:
            self.stopwords = stopwords.words("english")
        else:
            self.stopwords = custom_stopwords
        if split_timestamp:
            self.tweet["Timestamp"] = self.tweet_df.apply(
                lambda r: r.Date[11:20], axis=1
            )
            self.tweet["Date"] = self.tweet.apply(
                lambda r: parse(r.Date[4:11]).strftime("%d/%m/%Y"), axis=1
            )

    def process_tweets(self, lemmatiser, stemmer=None):

        if self.keyword is not None:
            row = list()
            for i in range(0, self.tweet_df.shape[0]):
                for words in self.keyword:
                    if bool(re.search(words, self.tweet_df["text"][i].lower())):
                        row.append(i)
            self.tweet_df.drop(self.tweet_df.index[row], inplace=True)

        self.__pre_processing()

        cleaned_tweets = []

        for data in self.tweet_df["Text"].values:

            tokenised = nltk.word_tokenize(str_no_punc)
            stop_free = [
                word for word in tokenised if word not in stopwords.words("english")
            ]
            word_lemm = [lemmatiser.lemmatize(t) for t in stop_free]
            if stemmer is not None:
                word_stem = [stemmer.stem(i) for i in word_lemm]
                cleaned_tweets.append(word_stem)
            else:
                cleaned_tweets.append(word_lemm)

        tweet["cleaned"] = cleaned_tweets
        return tweet

    def __removeusers(self, tweet_frame, exclude_users):
        row = list()
        for i in range(0, tweet_frame.shape[0]):
            if any(word in tweet_frame["User"][i] for word in exclude_users):
                row.append(i)

        tweet_frame.drop(tweet_frame.index[row], inplace=True)
        return tweet_frame

    def __remove_emoji(self):
        row = list()
        for i in range(0, tweet.shape[0]):
            if bool(re.search(emoji_pattern, self.tweet_df["text"][i])):
                row.append(i)
        return self.tweet_df.drop(self.tweet_df.index[row], inplace=True).reset_index(
            drop=True
        )

    def __pre_processing(self):
        str_lower = re.sub(
            r"https?:\/\/.*\/\w*", "", self.tweet_df["text"].values
        ).lower()
        str_letters_only = re.sub("[^a-zA-Z]", " ", str_lower)
        str_no_username = re.sub(r"(?:@[\w_]+)", "", str_letters_only).strip()
        return "".join(
            word for word in str_no_username if word not in set(string.punctuation)
        )

    def tokenised(self):
        pass

    def __stem_and_lemmatise(self):
        for word in words_list():
            stemmer.stem(lemmatiser.lemmatize(word))

        [lemmatiser.lemmatize(t) for t in stop_free]
        [stemmer.stem(i) for i in word_lemm]


class TweetParse(object):
    def __init__(self, token):
        self.consumer_key = token["consumer_key"]
        self.consumer_secret = token["consumer_secret"]
        self.access_token = token["access_token"]
        self.access_token_secret = token["access_token_secret"]

    def collect_tweets(self, keywords, metadata=None, mode="historical"):
        tweet_list = []
        if metadata is None:
            metadata = [
                "created_at",
                "screen_name",
                "favorite_count",
                "retweet_count",
                "followers_count",
                "friends_count",
                "text",
                "location",
            ]
        logger.info(" tweets")
        if mode == "historical":
            for tweet in self.historical_parser(keywords):
                print(
                    f"{tweet['created_at']} @ {tweet['user']['screen_name']} tweeted: {tweet['text']}"
                )
                tweet_list.append(
                    [
                        tweet["created_at"],
                        tweet["user"]["screen_name"],
                        tweet["favorite_count"],
                        tweet["retweet_count"],
                        tweet["user"]["followers_count"],
                        tweet["user"]["friends_count"],
                        tweet["text"],
                        tweet["user"]["location"],
                    ]
                )

        return pd.DataFrame(tweet_list, columns=metadata)

    def __historical_parser(self, keywords, include_entities=False):
        """
        Parses tweets which were posted in the past 10 days. Uses the
        TwitterSearch package.
        Parameters
        ----------
        keywords
        include_entities

        Returns
        -------

        """

        try:
            tso = TwitterSearchOrder()
            tso.set_language("en")
            tso.set_keywords(keywords)
            tso.set_include_entities(include_entities)
            ts = TwitterSearch(
                self.consumer_key,
                self.consumer_secret,
                self.access_token,
                self.access_token_secret,
            )
            tweet_gen = ts.search_tweets_iterable(tso)
        except TwitterSearchException as e:
            print(e)

        return tweet_gen

    @staticmethod
    def __streaming_parser():
        pass

    @classmethod
    def get_token_from_file(cls, token_path):
        """
        Provides alternative constructor for getting tokens
        from a text file
        Parameters
        ----------
        token_path

        Returns
        -------

        """
        token_dict = {}
        with open(token_path, "r") as fp:
            for line in fp:
                lst = line.strip().split(" ")
                token_dict[lst[0]] = lst[1]
        print(token_dict)
        return cls(token_dict)


if __name__ == "__main__":

    tweets = TweetParse.get_token_from_file("/Users/rk1103/Documents/tokens.txt")
    df = tweets.collect_tweets(keywords=["malaria"])
    print(df)
    # filter_users = tweets.removeusers(raw_tweets)
    # filtered_emojis = tweets.remove_emoji(filter_users, emoji_pattern)
    amend_dates = tweets.date_timestamp(filtered_emojis)
    stemmer = SnowballStemmer("english")
    lemmatiser = WordNetLemmatizer()
    cleaned_tweets = tweets.cleantweet(amend_dates, stemmer, lemmatiser)
