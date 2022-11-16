from typing import List, Dict
from nomic import AtlasClient

from datetime import datetime
import json

import tweepy

from dotenv import load_dotenv
import os

import snscrape.modules.twitter as sntwitter

from tqdm import tqdm

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
        self.atlas = AtlasClient
    
    def user_lookup(self, user: str, quantity: int):
        """Obtain Tweets from a specific user.

            Args:
                user(str): twitter handle of user
                quantity(int): the amount of tweets from the latest tweet needed
            Returns:
                a list of JSONs, of which each JSON represents all the metadata and data of each tweet

        """
        query: List[Dict] = []

        tweets = tweepy.Cursor(
            self.api.user_timeline, screen_name=user, count=200, tweet_mode="extended"
        ).items(quantity)  # tweepy.Cursor allows for pagination due to the single request tweet limitations

        for tweet in tweets:
            query.append(tweet._json)

        tweet_more = tweepy.Cursor(
                        self.api.user_timeline, max_id=query[-1], screen_name=user, count=200, tweet_mode="extended"
                    ).items(quantity)

        for tweet in tweet_more:
            query.append(tweet._json)

        return query

    def user_lookup_sns(self, user: str, quantity: int):
        """Obtain tweets from a specified user, using snsscrape

        :param user:
        :return:
        """
        query: List[Dict] = []

        for idx, tweet in enumerate(sntwitter.TwitterSearchScraper(f'from:{user}').get_items()):
            if idx > quantity:
                break
            query.append({"created_at": tweet.date, "tweet_id": tweet.id, "full_text": tweet.content, "user": tweet.user.username})

        return query

    
if __name__ == "__main__":
    bot = Utils()
    lookup = bot.user_lookup_sns("JoeBiden", 5000)
    print(len(lookup))
    print(lookup[-1])


