# import typing
import preprocessor as p
from nomic import AtlasClient
from utils import TwitterStream
import os
import jsonlines


class Profile:
    def __init__(self):
        self.twitter_stream = TwitterStream()
        self.atlas = AtlasClient()

    def create_social_profile(self, user: str):
        """Create social profile

        :param user: handle of twitter user that is used to create the social profile
        :return: Map link for social profile
        """
        lookup_amount = 5000
        tweets = [{"text": p.clean(tweet["full_text"])} for tweet in self.twitter_stream.user_lookup(user, lookup_amount)]
        for idx, tweet in enumerate(tweets):
            if len(tweet) < 20:
                tweets.pop(idx)
        # print(tweets[:5])

        if not os.path.exists(f'{user}_tweets.jsonl'):
            with jsonlines.open(f'{user}_tweets.jsonl', mode='a') as writer:
                for tweet in tweets:
                    writer.write(tweet)

        self.atlas.map_text(data=tweets,
                            indexed_field='text',
                            is_public=True,
                            map_name=f'Social Profile of the Latest Twitter CEO(Elon Musk) {len(tweets)}',
                            map_description=f"A {len(tweets)} tweet social profile of the latest Twitter CEO with Nomic's text embedder created by Yuvanesh Anand",
                            organization_name=None,  # defaults to your current user.
                            num_workers=10
                            )


if __name__ == "__main__":
    elon_musk_profiler = Profile()
    elon_musk_profiler.create_social_profile("elonmusk")
