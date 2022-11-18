from typing import List
import preprocessor as p
from nomic import AtlasClient
from utils import Utils
import jsonlines
from tqdm import tqdm


class Profile:
    def __init__(self):
        self.utils = Utils()
        self.atlas = AtlasClient()

    def create_social_profile_tweepy(self, map_name: str, map_description: str, users: List[str], outdir: str):
        """Create social profile with tweepy as tweet source

        :param map_name: name of atlas map
        :param map_description: description of atlas map
        :param users: handle of twitter user that is used to create the social profile
        :param outdir: specified directory of where the tweets(in JSON format) shoudl go
        """
        lookup_amount = 10000
        for user in users:
            tweets = [{"text": p.clean(tweet["full_text"]), "created_at": tweet["created_at"]} for tweet in self.utils.user_lookup(user, lookup_amount)]
            with jsonlines.open(f'{outdir}/{user}_tweets.jsonl', mode='a') as writer:
                for idx, tweet in enumerate(tweets):
                    if len(tweet["text"]) < 10:
                        tweets.pop(idx)
                        continue
                    writer.write(tweet)

        self.atlas.map_text(data=tweets,
                            indexed_field='text',
                            is_public=True,
                            map_name=map_name,
                            map_description=map_description,
                            organization_name=None,  # defaults to your current user.
                            num_workers=10
                            )

    def create_social_profile_sns(self, map_name: str, map_description: str, users: List[str], outdir: str):
        """

        :param map_name: name of atlas map
        :param map_description: description of atlas map
        :param users: handle of twitter user that is used to create the social profile
        :param outdir: specified directory of where the tweets(in JSON format) should go
        """
        all_tweets = []

        for user in tqdm(users):
            tweets = self.utils.user_lookup_sns(user, 10000)
            with jsonlines.open(f'{outdir}/{user}_tweets.jsonl', mode='a') as writer:
                for idx, tweet in enumerate(tweets):
                    if len(tweet["full_text"]) < 20 or None:
                        tweets.pop(idx)
                        continue

                    tweet["created_at"] = str(tweet["created_at"])
                    tweet["full_text"] = p.clean(tweet["full_text"])
                    all_tweets.append(tweet)
                    writer.write(tweet)

        self.atlas.map_text(data=all_tweets,
                            indexed_field='full_text',
                            is_public=True,
                            map_name=map_name,
                            map_description=map_description,
                            colorable_fields=["user"],
                            )


if __name__ == "__main__":
    profiler = Profile()
    profiler.create_social_profile_sns(outdir='data/', map_name='Social Profile of the current POTUS', map_description="A social profile of the latest POTUS Joe Biden, with Nomic's text embedder created by Yuvanesh Anand", users=["JoeBiden", "POTUS"])
