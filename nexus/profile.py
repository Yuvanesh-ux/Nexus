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
import re
from dotenv import load_dotenv

load_dotenv()

# ===== Resource Limiting Constants =====
MAX_TOTAL_TWEETS = 10000  # Max tweets to embed/cluster in one run (total, all users)
MAX_TWEETS_PER_USER = 3000  # Max tweets loaded per user


class Profile:
    def __init__(self):
        self.utils = Utils()
        self.atlas = AtlasClient()

    @staticmethod
    def _confirm_publication() -> bool:
        print(
            "\nWARNING: You are about to publish all collected tweets and enriched metadata to the public Atlas instance."
            "\nThis may expose sensitive analytical metadata (e.g., clustering, timestamps, user handles) that users did NOT consent to republishing or deanonymising."
            "\nDo you wish to continue and make this map PUBLIC? (yes/[no])"
        )
        response = input("> ").strip().lower()
        if response in ["yes", "y"]:
            print("Proceeding: Map will be public.")
            return True
        else:
            print("Aborting public publishing: Map will be kept PRIVATE.")
            return False

    def _sanitize_outdir(self, outdir: str) -> str:
        # Ensure outdir is a canonical absolute path and exists
        outdir_abs = os.path.abspath(outdir)
        if not os.path.isdir(outdir_abs):
            raise ValueError(f"Output directory '{outdir}' does not exist or is not a directory.")
        return outdir_abs

    def _sanitize_username(self, username: str) -> str:
        # Keep only alphanumerics, underscores, dashes, and periods for filenames
        sanitized = re.sub(r'[^A-Za-z0-9_\-\.]', '', username)
        return sanitized

    def _safe_join(self, base: str, *paths) -> str:
        # Canonicalize path and check it's still within base
        abs_base = os.path.abspath(base)
        result_path = os.path.abspath(os.path.join(abs_base, *paths))
        if os.path.commonpath([abs_base, result_path]) != abs_base:
            raise ValueError("Attempted path traversal detected in path construction.")
        return result_path

    def create_social_profile_tweepy(self, map_name: str, map_description: str, users: List[str], outdir: str):
        """Create social profile with tweepy as tweet source

        :param map_name: name of atlas map
        :param map_description: description of atlas map
        :param users: handle of twitter user that is used to create the social profile
        :param outdir: specified directory of where the tweets(in JSON format) shoudl go
        """
        outdir_safe = self._sanitize_outdir(outdir)
        lookup_amount = min(10000, MAX_TWEETS_PER_USER)
        for user in users:
            username_safe = self._sanitize_username(user)
            if not username_safe:
                logger.warning(f"Username '{user}' was sanitized to empty string, skipping.")
                continue
            tweets = [{"text": p.clean(tweet["full_text"]), "created_at": tweet["created_at"]} for tweet in
                      self.utils.user_lookup(user, lookup_amount)]
            # Limit and filter tweets
            filtered_tweets = []
            outfile_path = self._safe_join(outdir_safe, f"{username_safe}_tweets.jsonl")
            with jsonlines.open(outfile_path, mode='a') as writer:
                for idx, tweet in enumerate(tweets):
                    if len(tweet["text"]) < 10:
                        continue
                    filtered_tweets.append(tweet)
                    writer.write(tweet)
                if len(filtered_tweets) > MAX_TWEETS_PER_USER:
                    logger.warning(f"User '{user}': truncating tweets to MAX_TWEETS_PER_USER ({MAX_TWEETS_PER_USER}).")
                    filtered_tweets = filtered_tweets[:MAX_TWEETS_PER_USER]
            # for map_text use only the (potentially truncated) filtered_tweets
            tweets = filtered_tweets

        # Runtime confirmation before publishing to public Atlas
        is_public = self._confirm_publication()

        self.atlas.map_text(data=tweets,
                            indexed_field='text',
                            is_public=is_public,
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
        if outdir:
            outdir_safe = self._sanitize_outdir(outdir)
        else:
            outdir_safe = ''
        all_tweets = []
        total_tweets_added = 0

        for user in tqdm(users):
            username_safe = self._sanitize_username(user)
            if not username_safe:
                logger.warning(f"Username '{user}' was sanitized to empty string, skipping.")
                continue
            user_tweets_temp = []
            try:
                logger.info(f"Loading {user}'s tweets from disk")
                if outdir_safe:
                    data_path = self._safe_join(outdir_safe, f"{username_safe}_tweets.jsonl")
                else:
                    data_path = f"{username_safe}_tweets.jsonl"
                with jsonlines.open(data_path, mode="r") as tweets:
                    for tweet in tweets:
                        if len(tweet.get("full_text", tweet.get("text", ""))) > 30:
                            user_tweets_temp.append(tweet)
                        if len(user_tweets_temp) >= MAX_TWEETS_PER_USER:
                            logger.warning(f"User '{user}': loaded MAX_TWEETS_PER_USER ({MAX_TWEETS_PER_USER}) from disk.")
                            break
            except BaseException:
                logger.info(f"Not on disk! scraping {user}'s tweets now")
                tweets = self.utils.user_lookup_sns(user, min(10000, MAX_TWEETS_PER_USER))
                if outdir_safe:
                    outfile_path = self._safe_join(outdir_safe, f"{username_safe}_tweets.jsonl")
                else:
                    outfile_path = f"{username_safe}_tweets.jsonl"
                with jsonlines.open(outfile_path, mode='a') as writer:
                    for idx, tweet in enumerate(tweets):
                        tweet["full_text"] = p.clean(tweet["full_text"])
                        if len(tweet["full_text"]) > 30:
                            tweet["created_at"] = str(tweet["created_at"])
                            user_tweets_temp.append(tweet)
                            writer.write(tweet)
                        if len(user_tweets_temp) >= MAX_TWEETS_PER_USER:
                            logger.warning(f"User '{user}': wrote MAX_TWEETS_PER_USER ({MAX_TWEETS_PER_USER}) to disk.")
                            break
            # Append user tweets to main collection, checking overall resource limit
            tweets_to_add = min(MAX_TOTAL_TWEETS - total_tweets_added, len(user_tweets_temp))
            if tweets_to_add < len(user_tweets_temp):
                logger.warning(f"Only {tweets_to_add} out of {len(user_tweets_temp)} tweets loaded for user '{user}' due to MAX_TOTAL_TWEETS ({MAX_TOTAL_TWEETS}) constraint.")
            all_tweets.extend(user_tweets_temp[:tweets_to_add])
            total_tweets_added += tweets_to_add
            if total_tweets_added >= MAX_TOTAL_TWEETS:
                logger.warning(f"Reached MAX_TOTAL_TWEETS ({MAX_TOTAL_TWEETS}). Skipping remaining tweets/users.")
                break

        for idx, tweet in enumerate(all_tweets):
            tweet["id"] = str(idx)

        if topics:
            n_cluster_docs = [40]
            for n_clusters in n_cluster_docs:
                logger.info(f"computing {n_clusters} cluster layer")
                try:
                    with open(f"data/cluster_labels/{self._sanitize_username(users[0])}_id_to_cluster_label_{n_clusters}", "r") as f:
                        id_to_cluster_label = json.load(f)
                    logger.info("Loaded all resources from disk")
                    print(id_to_cluster_label[-1])
                except BaseException:
                    # remake clusters
                    id_to_cluster_label = {}

                    try:
                        logger.info("Loading embeddings from disk.")
                        embeddings = np.load(embedding_path)
                        if len(embeddings) > MAX_TOTAL_TWEETS:
                            logger.warning(f"Embeddings shape {embeddings.shape[0]} exceeds MAX_TOTAL_TWEETS ({MAX_TOTAL_TWEETS}). Truncating.")
                            embeddings = embeddings[:MAX_TOTAL_TWEETS]
                    except BaseException:
                        logger.info("Embedding with Cohere")
                        cohere_api_key = os.getenv("COHERE_KEY")
                        tweet_texts = [datum['full_text'] for datum in all_tweets]
                        if len(tweet_texts) > MAX_TOTAL_TWEETS:
                            logger.warning(f"Truncating embedding input from {len(tweet_texts)} to MAX_TOTAL_TWEETS ({MAX_TOTAL_TWEETS}).")
                            tweet_texts = tweet_texts[:MAX_TOTAL_TWEETS]
                            all_tweets = all_tweets[:MAX_TOTAL_TWEETS]
                        embedder = CohereEmbedder(cohere_api_key=cohere_api_key)
                        embeddings = np.array(embedder.embed(texts=tweet_texts)).squeeze()
                        with open(embedding_path, 'wb') as f:
                            np.save(f, embeddings)
                    logger.info("Running Kmeans to generate clusters")
                    if len(embeddings) > MAX_TOTAL_TWEETS:
                        logger.warning(f"Further truncating embeddings from shape {embeddings.shape} to length MAX_TOTAL_TWEETS ({MAX_TOTAL_TWEETS}).")
                        embeddings = embeddings[:MAX_TOTAL_TWEETS]
                    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
                    for datum, cluster_id in zip(all_tweets, [int(i) for i in list(kmeans.labels_)]):
                        id_to_cluster_label[datum['id']] = cluster_id

                    with open(f'data/cluster_labels/{self._sanitize_username(users[0])}_id_to_cluster_label_{n_clusters}', 'w') as f:
                        json.dump(id_to_cluster_label, f)
                print(len(all_tweets))
                logger.info("Computing Topics")
                cluster_rarity_list = self.utils.create_topics(all_tweets, id_to_cluster_label=id_to_cluster_label)

                for idx, datum in enumerate(all_tweets):
                    datum_cluster = id_to_cluster_label[str(idx)]
                    datum[f"cluster_{n_clusters}"] = datum_cluster
                    datum[f"topic_{n_clusters}"] = cluster_rarity_list[datum_cluster][1]

        # Runtime confirmation before publishing to public Atlas
        is_public = self._confirm_publication()

        self.atlas.map_text(data=all_tweets,
                            indexed_field='full_text',
                            is_public=is_public,
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