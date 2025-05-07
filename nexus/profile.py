from typing import List, Optional
import preprocessor as p
from nomic import AtlasClient, CohereEmbedder
from utils import Utils
import jsonlines
import json
from tqdm import tqdm
from loguru import logger
import numpy as np
from sklearn.cluster import KMeans
import os
from dotenv import load_dotenv
import re

load_dotenv()


class Profile:
    def __init__(self):
        self.utils = Utils()
        self.atlas = AtlasClient()

    def _is_valid_tweet(self, tweet: dict, min_length: int = 30, max_length: int = 512) -> bool:
        """
        Perform validation/sanity checks on ingested tweets to reduce data poisoning risk.
        - Enforces min/max length,
        - Checks for only-printable UTF-8,
        - Rejects tweets with overwhelming URLs, control chars, or binary data,
        - Basic anomaly: extremely repetitive/unusual text.
        """
        text = tweet.get("full_text") or tweet.get("text")
        if not text or not isinstance(text, str):
            return False

        # Length checks
        if len(text) < min_length or len(text) > max_length:
            return False

        # Attempt to encode as UTF-8 and check printable chars
        try:
            text.encode("utf-8")
        except Exception:
            return False

        # Reject text containing null bytes or lots of control chars
        if "\x00" in text or len([c for c in text if ord(c) < 32 and c not in "\n\r\t"]) > 3:
            return False

        # Limit urls
        url_count = len(re.findall(r'https?://\S+', text))
        if url_count > 3:
            return False

        # High unicode/surrogates
        if len([c for c in text if ord(c) > 0xFFFF]) > 5:
            return False

        # Very repetitive or single-character spam (anomaly)
        if len(set(text.strip())) < 4 and len(text) > 40:
            return False

        # For tweets that look like "RT RT RT... [50x]"
        if re.match(r'^(\w{1,5})( \1){6,}', text):
            return False

        # Passed all checks
        return True

    def _deduplicate_tweets(self, tweets: List[dict]) -> List[dict]:
        """Deduplicate tweets by full_text/content and created_at field."""
        seen = set()
        unique = []
        for tweet in tweets:
            text = tweet.get("full_text") or tweet.get("text")
            created_at = tweet.get("created_at", "")
            k = (text, created_at)
            if k not in seen:
                seen.add(k)
                unique.append(tweet)
        return unique

    def create_social_profile_tweepy(self, map_name: str, map_description: str, users: List[str], outdir: str):
        """Create social profile with tweepy as tweet source

        :param map_name: name of atlas map
        :param map_description: description of atlas map
        :param users: handle of twitter user that is used to create the social profile
        :param outdir: specified directory of where the tweets(in JSON format) shoudl go
        """
        lookup_amount = 10000
        for user in users:
            tweets = []
            for tweet in self.utils.user_lookup(user, lookup_amount):
                cleaned_text = p.clean(tweet["full_text"])
                record = {
                    "text": cleaned_text,
                    "created_at": tweet["created_at"]
                }
                # Apply validation
                if not self._is_valid_tweet(record, min_length=10):
                    continue
                record["_provenance"] = "scraped"
                tweets.append(record)

            tweets = self._deduplicate_tweets(tweets)
            with jsonlines.open(f'{outdir}/{user}_tweets.jsonl', mode='a') as writer:
                for tweet in tweets:
                    writer.write(tweet)

        self.atlas.map_text(data=tweets,
                            indexed_field='text',
                            is_public=True,
                            map_name=map_name,
                            map_description=map_description,
                            organization_name=None,  # defaults to your current user.
                            num_workers=10
                            )

    def create_social_profile_sns(self,
                                  map_name: str,
                                  map_description: str,
                                  users: List[str],
                                  outdir: Optional[str] = '',
                                  topics: bool = False,
                                  embedding_path: str = ''
    ):
        """

        :param embedding_path: path of npy file for topic extraction
        :param topics: indicating whether or not you want to automatic topic extraction
        :param map_name: name of atlas map
        :param map_description: description of atlas map
        :param users: handle of twitter user that is used to create the social profile
        :param outdir: specified directory of where the tweets(in JSON format) should go
        """
        all_tweets = []

        for user in tqdm(users):
            found_valid_on_disk = False
            try:
                logger.info(f"Loading {user}'s tweets from disk")
                data_path = os.path.join(outdir, f"{user}_tweets.jsonl")
                with jsonlines.open(data_path, mode="r") as tweets:
                    for tweet in tweets:
                        # Ensure every tweet loaded from disk is validated and cleaned!
                        if "full_text" in tweet and tweet["full_text"]:
                            tweet["full_text"] = p.clean(tweet["full_text"])
                        elif "text" in tweet:
                            tweet["full_text"] = p.clean(tweet["text"])
                        else:
                            continue
                        if not self._is_valid_tweet(tweet):
                            logger.warning(f"Invalid or anomalous tweet from disk for user {user}, skipping.")
                            continue
                        tweet["_provenance"] = "disk"
                        tweet["user"] = user
                        all_tweets.append(tweet)
                        found_valid_on_disk = True
            except BaseException:
                logger.info(f"Not on disk! scraping {user}'s tweets now")

            if not found_valid_on_disk:
                # Scrape tweets
                tweets = self.utils.user_lookup_sns(user, 10000)
                scraped_records = []
                for tweet in tweets:
                    cleaned_text = p.clean(tweet["full_text"])
                    record = {
                        "full_text": cleaned_text,
                        "created_at": str(tweet["created_at"]),
                        "_provenance": "scraped",
                        "user": user
                    }
                    if not self._is_valid_tweet(record):
                        continue
                    scraped_records.append(record)
                    all_tweets.append(record)
                scraped_records = self._deduplicate_tweets(scraped_records)
                with jsonlines.open(f'{outdir}/{user}_tweets.jsonl', mode='a') as writer:
                    for tweet in scraped_records:
                        writer.write(tweet)

        # Deduplicate corpus
        all_tweets = self._deduplicate_tweets(all_tweets)

        for idx, tweet in enumerate(all_tweets):
            tweet["id"] = str(idx)

        if topics:
            n_cluster_docs = [40]
            for n_clusters in n_cluster_docs:
                logger.info(f"computing {n_clusters} cluster layer")
                try:
                    with open(f"data/cluster_labels/{users[0]}_id_to_cluster_label_{n_clusters}", "r") as f:
                        id_to_cluster_label = json.load(f)
                    logger.info("Loaded all resources from disk")
                    print(id_to_cluster_label[-1])
                except BaseException:
                    # remake clusters
                    id_to_cluster_label = {}

                    try:
                        logger.info("Loading embeddings from disk.")
                        embeddings = np.load(embedding_path)
                    except BaseException:
                        logger.info("Embedding with Cohere")
                        cohere_api_key = os.getenv("COHERE_KEY")
                        embedder = CohereEmbedder(cohere_api_key=cohere_api_key)
                        embeddings = np.array(embedder.embed(texts=[datum['full_text'] for datum in all_tweets])).squeeze()
                        with open(embedding_path, 'wb') as f:
                            np.save(f, embeddings)
                    logger.info("Running Kmeans to generate clusters")
                    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
                    for datum, cluster_id in zip(all_tweets, [int(i) for i in list(kmeans.labels_)]):
                        id_to_cluster_label[datum['id']] = cluster_id

                    with open(f'data/cluster_labels/{users[0]}_id_to_cluster_label_{n_clusters}', 'w') as f:
                        json.dump(id_to_cluster_label, f)
                print(len(all_tweets))
                logger.info("Computing Topics")
                cluster_rarity_list = self.utils.create_topics(all_tweets, id_to_cluster_label=id_to_cluster_label)

                for idx, datum in enumerate(all_tweets):
                    datum_cluster = id_to_cluster_label[str(idx)]
                    datum[f"cluster_{n_clusters}"] = datum_cluster
                    datum[f"topic_{n_clusters}"] = cluster_rarity_list[datum_cluster][1]

        self.atlas.map_text(data=all_tweets,
                            indexed_field='full_text',
                            is_public=True,
                            map_name=map_name,
                            map_description=map_description,
                            colorable_fields=["topic_40","user"],
                            )


if __name__ == "__main__":
    profiler = Profile()
    profiler.create_social_profile_sns(outdir='data/',
                                       map_name='Social Profile of the current POTUS',
                                       map_description="A social profile of the latest POTUS Joe Biden, with Nomic's text embedder created by Yuvanesh Anand",
                                       users=["JoeBiden", "POTUS"],
                                       topics=True,
                                       embedding_path="embeddings/JoeBiden.npy")