from typing import List
import preprocessor as p
from nomic import AtlasClient
from utils import Utils
import os
import jsonlines
import datetime
from tqdm import tqdm

class Profile:
    def __init__(self):
        self.utils = Utils()
        self.atlas = AtlasClient()

    def create_social_profile(self, social_profile_map_name: str, users: List[str]):
        """Create social profile

        :param users: handle of twitter user that is used to create the social profile
        :return: Map link for social profile
        """
        lookup_amount = 10000
        for user in users:
            tweets = [{"text": p.clean(tweet["full_text"]), "created_at": tweet["created_at"]} for tweet in self.utils.user_lookup(user, lookup_amount)]
            with jsonlines.open(f'{social_profile_map_name}_tweets.jsonl', mode='a') as writer:
                for idx, tweet in enumerate(tweets):
                    if len(tweet["text"]) < 10:
                        tweets.pop(idx)
                        continue
                    writer.write(tweet)

        # if not os.path.exists(f'{user}_tweets.jsonl'):
        #     with jsonlines.open(f'{user}_tweets.jsonl', mode='a') as writer:
        #         for tweet in tweets:
        #             writer.write(tweet)

        self.atlas.map_text(data=tweets,
                            indexed_field='text',
                            is_public=True,
                            map_name=f'Social Profile of the current POTUS(Joe Biden) {len(tweets)}',
                            map_description=f"A {len(tweets)} tweet social profile of the current president of the united states, Joe Biden, with Nomic's text embedder created by Yuvanesh Anand",
                            organization_name=None,  # defaults to your current user.
                            num_workers=10
                            )

    def create_social_profile_sns(self, social_profile_map_name: str, users: List[str]):
        all_tweets = []
        for user in tqdm(users):
            tweets = self.utils.user_lookup_sns(user, 5000)
            with jsonlines.open(f'{social_profile_map_name}_tweets.jsonl', mode='a') as writer:
                for idx, tweet in enumerate(tweets):
                    if len(tweet["full_text"]) < 10:
                        tweets.pop(idx)
                        continue

                    tweet["created_at"] = str(tweet["created_at"])
                    all_tweets.append(tweet)
                    writer.write(tweet)


        self.atlas.map_text(data=all_tweets,
                            indexed_field='full_text',
                            is_public=True,
                            map_name=f'Social Profile of the current POTUS(Joe Biden) {len(all_tweets)}',
                            map_description=f"A {len(all_tweets)} tweet social profile of the current president of the united states, Joe Biden, with Nomic's text embedder created by Yuvanesh Anand",
                            colorable_fields=["user"],
                            organization_name=None,  # defaults to your current user.
                            num_workers=10
                            )


if __name__ == "__main__":
    elon_musk_profiler = Profile()
    elon_musk_profiler.create_social_profile_sns("Joe Biden", ["JoeBiden","POTUS"])
