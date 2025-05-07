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

load_dotenv()


class Profile:
    def __init__(self):
        self.utils = Utils()
        self.atlas = AtlasClient()
        # Define safe base directories (relative to current working dir)
        self.safe_outdir_base = os.path.abspath("data")
        self.safe_embedding_base = os.path.abspath("embeddings")

    def _safe_join(self, base_dir: str, *paths: str) -> str:
        """
        Safely join one or more path components to the base_dir, ensuring the result
        does not escape the base_dir (prevents path traversal).
        """
        joined_path = os.path.abspath(os.path.join(base_dir, *paths))
        if not joined_path.startswith(base_dir + os.sep) and joined_path != base_dir:
            raise ValueError(f"Unsafe path detected: {joined_path} escapes base directory {base_dir}")
        return joined_path

    def _sanitize_username(self, user: str) -> str:
        # Block users with path components, special chars, or traversal
        if "/" in user or "\\" in user or ".." in user or user.startswith(".") or user == "":
            raise ValueError(f"Invalid username: {user!r}")
        return user

    def _sanitize_outdir(self, outdir: str) -> str:
        # Only allow paths inside self.safe_outdir_base
        outdir_abs = os.path.abspath(outdir)
        if not outdir_abs.startswith(self.safe_outdir_base + os.sep) and outdir_abs != self.safe_outdir_base:
            raise ValueError(f"Unsafe outdir specified: {outdir!r}")
        return outdir_abs

    def _sanitize_embedding_path(self, embedding_path: str) -> str:
        # Only allow embeddings within self.safe_embedding_base, and .npy files
        # Not allow path traversal/absolute
        if not embedding_path:
            raise ValueError("embedding_path cannot be empty")
        if os.path.isabs(embedding_path):
            raise ValueError("Absolute embedding_path not allowed")
        if ".." in embedding_path or embedding_path.startswith("/") or embedding_path.startswith("\\"):
            raise ValueError("Path traversal in embedding_path not allowed")
        if not embedding_path.endswith(".npy"):
            raise ValueError("Embedding path must end with .npy extension")
        # Force to safe base
        full_path = os.path.abspath(os.path.join(self.safe_embedding_base, embedding_path))
        if not full_path.startswith(self.safe_embedding_base + os.sep):
            raise ValueError("embedding_path escapes embeddings directory")
        return full_path

    def create_social_profile_tweepy(self, map_name: str, map_description: str, users: List[str], outdir: str):
        """Create social profile with tweepy as tweet source

        :param map_name: name of atlas map
        :param map_description: description of atlas map
        :param users: handle of twitter user that is used to create the social profile
        :param outdir: specified directory of where the tweets(in JSON format) shoudl go
        """
        outdir_safe = self._sanitize_outdir(outdir)
        lookup_amount = 10000
        for user in users:
            sanitized_user = self._sanitize_username(user)
            tweets = [{"text": p.clean(tweet["full_text"]), "created_at": tweet["created_at"]} for tweet in
                      self.utils.user_lookup(sanitized_user, lookup_amount)]
            tweet_file = self._safe_join(outdir_safe, f'{sanitized_user}_tweets.jsonl')
            with jsonlines.open(tweet_file, mode='a') as writer:
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

        # Validate/sanitize outdir and embedding_path early
        outdir_safe = self._sanitize_outdir(outdir or self.safe_outdir_base)
        embedding_full_path = None
        if embedding_path:
            embedding_full_path = self._sanitize_embedding_path(embedding_path)

        for user in tqdm(users):
            try:
                sanitized_user = self._sanitize_username(user)
                logger.info(f"Loading {sanitized_user}'s tweets from disk")
                data_path = self._safe_join(outdir_safe, f"{sanitized_user}_tweets.jsonl")
                with jsonlines.open(data_path, mode="r") as tweets:
                    for tweet in tweets:
                        all_tweets.append(tweet)
            except BaseException:
                logger.info(f"Not on disk! scraping {user}'s tweets now")
                sanitized_user = self._sanitize_username(user)
                tweets = self.utils.user_lookup_sns(sanitized_user, 10000)
                tweet_file = self._safe_join(outdir_safe, f"{sanitized_user}_tweets.jsonl")
                with jsonlines.open(tweet_file, mode='a') as writer:
                    for idx, tweet in enumerate(tweets):
                        tweet["full_text"] = p.clean(tweet["full_text"])
                        if len(tweet["full_text"]) > 30:
                            tweet["created_at"] = str(tweet["created_at"])
                            all_tweets.append(tweet)
                            writer.write(tweet)

            for idx, tweet in enumerate(all_tweets):
                tweet["id"] = str(idx)

        if topics:
            n_cluster_docs = [40]
            for n_clusters in n_cluster_docs:
                logger.info(f"computing {n_clusters} cluster layer")
                try:
                    cluster_label_dir = os.path.abspath("data/cluster_labels")
                    os.makedirs(cluster_label_dir, exist_ok=True)
                    cluster_file = self._safe_join(cluster_label_dir, f"{self._sanitize_username(users[0])}_id_to_cluster_label_{n_clusters}")
                    with open(cluster_file, "r") as f:
                        id_to_cluster_label = json.load(f)
                    logger.info("Loaded all resources from disk")
                    print(id_to_cluster_label[-1])
                except BaseException:
                    # remake clusters
                    id_to_cluster_label = {}

                    try:
                        logger.info("Loading embeddings from disk.")
                        if not embedding_full_path:
                            raise ValueError("Embedding path is required for topic extraction")
                        embeddings = np.load(embedding_full_path)
                    except BaseException:
                        logger.info("Embedding with Cohere")
                        cohere_api_key = os.getenv("COHERE_KEY")
                        embedder = CohereEmbedder(cohere_api_key=cohere_api_key)
                        embeddings = np.array(embedder.embed(texts=[datum['full_text'] for datum in all_tweets])).squeeze()
                        if not embedding_full_path:
                            raise ValueError("Embedding path is required for saving new embedding")
                        # Ensure embedding directory exists
                        os.makedirs(os.path.dirname(embedding_full_path), exist_ok=True)
                        with open(embedding_full_path, 'wb') as f:
                            np.save(f, embeddings)
                    logger.info("Running Kmeans to generate clusters")
                    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
                    for datum, cluster_id in zip(all_tweets, [int(i) for i in list(kmeans.labels_)]):
                        id_to_cluster_label[datum['id']] = cluster_id

                    # Save cluster labels file safely
                    cluster_label_dir = os.path.abspath("data/cluster_labels")
                    os.makedirs(cluster_label_dir, exist_ok=True)
                    cluster_file = self._safe_join(cluster_label_dir, f"{self._sanitize_username(users[0])}_id_to_cluster_label_{n_clusters}")
                    with open(cluster_file, 'w') as f:
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
                                       embedding_path="JoeBiden.npy")