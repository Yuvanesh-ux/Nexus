from typing import List, Dict

from datetime import datetime
import json

import tweepy

from dotenv import load_dotenv
import os

load_dotenv()

class TwitterStream:
    def __init__(self):
        api_key = os.getenv("API_KEY")
        api_secret = os.getenv("API_SECRET")
        access_token = os.getenv("ACCESS_TOKEN")
        access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")

        auth = tweepy.OAuthHandler(api_key, api_secret)
        auth.set_access_token(access_token, access_token_secret)

        self.api = tweepy.API(auth)
    
    def user_lookup(self, user: str, quantity: int):
        """Obtain Tweets from a specific user.

            Args:
                user(str): twitter handle of user
                limit(int): the amount of tweets from the latest tweet needed
            Returns:
                a list of JSONs, of which each JSON represents all the metadata and data of each tweet

        """

        tweets = tweepy.Cursor(
            self.api.user_timeline, screen_name=user, count=200, tweet_mode="extended"
        ).items(quantity)# tweepy.Cursor allows for pagination due to the single request tweet limitations

        temp_list: List[Dict] = []

        for tweet in tweets:
            temp_list.append(tweet._json)

        return temp_list
    
if __name__ == "__main__":
    bot = TwitterStream()
    lookup = bot.user_lookup("elonmusk", 10000)
    print(len(lookup))
    print(lookup[-1]["full_text"])


