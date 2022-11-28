import tweepy
from dotenv import load_dotenv
import os
import snscrape.modules.twitter as sntwitter
from tqdm import tqdm
from loguru import logger
from collections import defaultdict
from nltk.corpus import stopwords
from typing import List, Dict, Tuple
import string
import re
stop_words = set(stopwords.words('english'))

load_dotenv()

class Utils:
    def __init__(self):
        api_key = os.getenv("API_KEY")
        api_secret = os.getenv("API_SECRET")
        access_token = os.getenv("ACCESS_TOKEN")
        access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")

        auth = tweepy.OAuthHandler(api_key, api_secret)
        auth.set_access_token(access_token, access_token_secret)

        self.api = tweepy.API(auth)
    
    def user_lookup_tweepy(self, user: str, quantity: int):
        """Obtain Tweets from a specific user.(only up to 3200 tweets)

        :param user: specified user you want to retrieve tweets from
        :param quantity: amount of tweets you want to retrieve

        """
        query: List[Dict] = []

        tweets = tweepy.Cursor(
            self.api.user_timeline, screen_name=user, count=200, tweet_mode="extended"
        ).items(quantity)  # tweepy.Cursor allows for pagination due to the single request tweet limitations

        for tweet in tweets:
            query.append(tweet._json)

        return query

    def user_lookup_sns(self, user: str, quantity: int):
        """Obtain tweets from a specified user, using snsscrape

        :param quantity: last x tweets needed, chronologically
        :param user: twitter handle of user
        """
        query: List[Dict] = []

        logger.info(f"Pulling {user}'s tweets")
        for idx, tweet in tqdm(enumerate(sntwitter.TwitterSearchScraper(f'from:{user}').get_items())):
            if idx > quantity:
                break
            query.append({"full_text": tweet.content, "tweet_link": f"https://twitter.com/{tweet.user.username}/status/{tweet.id}" , "created_at": tweet.date, "tweet_id": tweet.id, "user": tweet.user.username})

        return query

    def create_topics(self, documents: List[Dict], id_to_cluster_label: Dict, id_field='id', text_field='full_text'):
        """
            A fast implementation of cluster n-gram frequency inverse corpus n-gram frequency
            n-grams cannot be a cluster label if:
            1. They contain stopwords
            2. They contain digits.
            """
        ngram_corpus_frequency = defaultdict(lambda: 0)
        ngram_cluster_frequency = defaultdict(lambda: defaultdict(lambda: 0))
        ngram_rarity_cluster_list = []

        documents.pop(-1)

        for datum in documents:
            cluster_id = id_to_cluster_label[datum[id_field]]

            text = datum[text_field].translate(str.maketrans('', '', string.punctuation))
            tokens = re.split(r'\W+', text)

            ngrams_in_document = set()
            for i in range(len(tokens) - 1):
                unigram = tokens[i]
                bigram = " ".join(tokens[i:i + 2])

                if tokens[i] in stop_words or tokens[i + 1] in stop_words:
                    continue

                unigram_contains_digits = any(j.isdigit() for j in tokens[i])
                bigram_contains_digits = unigram_contains_digits or any(j.isdigit() for j in tokens[i + 1])

                # ngrams only count if they do not contain digits.
                # If an ngram appears multiple times in a document, count it only once.

                if not unigram_contains_digits and not unigram in ngrams_in_document:
                    ngram_corpus_frequency[unigram] += 1
                    ngram_cluster_frequency[cluster_id][unigram] += 1
                    ngrams_in_document.add(unigram)

                # if bigram.isalpha():
                if not bigram_contains_digits and not bigram in ngrams_in_document:
                    ngram_corpus_frequency[bigram] += 1
                    ngram_cluster_frequency[cluster_id][bigram] += 1
                    ngrams_in_document.add(bigram)
            ngrams_in_document = set()

        for cluster_id, ngram_frequencies in ngram_cluster_frequency.items():
            ngram_rarity: List[Tuple] = []
            for ngram, cluster_frequency in ngram_frequencies.items():
                if cluster_frequency > 5:  # ensure they are not just rare tokens in the corpus, but do occur with some freqency in the clusters
                    rarity = float(cluster_frequency) / float(ngram_corpus_frequency[ngram])
                    ngram_rarity.append((ngram, rarity))
            try:
                rarest_ngrams = sorted(ngram_rarity, key=lambda item: item[1], reverse=True)[
                                :5]  # [0][0] #first [0] is to get the tuple, second [0] is to access the first element of the tuple
                rarest_ngram = sorted(rarest_ngrams, key=len, reverse=True)[0][0]
            except IndexError:
                print("empty abstract")

            print(cluster_id)
            print(rarest_ngrams)
            print()

            ngram_rarity_cluster_list.append((cluster_id, rarest_ngram))

        ngram_rarity_cluster_list.sort(key=lambda item: item[0])
        print(ngram_rarity_cluster_list)
        return ngram_rarity_cluster_list  # [0] is cluster, [1] is topic label


    
if __name__ == "__main__":
    bot = Utils()
    lookup = bot.user_lookup_sns("JoeBiden", 5000)
    print(len(lookup))
    print(lookup[-1])


