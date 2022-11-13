# import typing

from nomic import AtlasClient
from utils import TwitterStream
import os
import jsonlines

class Profile:
    def __init__(self):
        self.twitter_stream = TwitterStream()
        self.atlas = AtlasClient()

    def social_profile(self, user: str):
        tweets = [tweet["full_text"] for tweet in self.twitter_stream.user_lookup(user, 1000)]

        if not os.path.exists(f'{user}_tweets.jsonl'):
            with jsonlines.open(f'{user}_tweets.jsonl', mode='a') as writer:
                for tweet in tweets:
                    dict = {}
                    dict["text"] = tweet
                    writer.write(dict)

        self.atlas.map_text(data=tweets,
                          indexed_field='text',
                          is_public=True,
                          map_name='Social Profile of the Latest Twitter CEO',
                          map_description="A 1000 tweet social profile of the latest Twitter CEO with Nomic's text embedder.",
                          organization_name = None,  # defaults to your current user.
                          num_workers = 10
                        )

if __name__ == "__main__":
    elon_musk_profiler = Profile()
    elon_musk_profiler.social_profile("elonmusk")

        
