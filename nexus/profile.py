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


def safe_filename(s: str) -> str:
    """Sanitize a string to be safe for filenames: allow [a-zA-Z0-9_.-], replace others with _."""
    return re.sub(r'[^a-zA-Z0-9_\.\-]', '_', s)

def _safe_join(base: str, *paths) -> str:
    """
    Safely join one or more path components to the base directory, preventing path traversal.
    Returns the joined absolute path, or raises ValueError if traversal detected.
    """
    base_abspath = os.path.abspath(base)
    path = os.path.abspath(os.path.join(base_abspath, *paths))
    if not os.path.commonpath([base_abspath, path]) == base_abspath:
        raise ValueError("Detected path traversal attempt (outside configured directory)")
    return path

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
        # Sanitize and normalize outdir
        outdir = os.path.abspath(outdir) if outdir else os.path.abspath("data")
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
        for user in users:
            sanitized_user = safe_filename(user)
            tweets = [{"text": p.clean(tweet["full_text"]), "created_at": tweet["created_at"]} for tweet in
                      self.utils.user_lookup(user, lookup_amount)]
            tweet_file_path = _safe_join(outdir, f'{sanitized_user}_tweets.jsonl')
            with jsonlines.open(tweet_file_path, mode='a') as writer:
                for idx, tweet in enumerate(tweets[:]):
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

        # Sanitize and normalize outdir
        outdir = os.path.abspath(outdir) if outdir else os.path.abspath("data")
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)

        for user in tqdm(users):
            sanitized_user = safe_filename(user)
            try:
                logger.info(f"Loading {user}'s tweets from disk")
                data_path = _safe_join(outdir, f"{sanitized_user}_tweets.jsonl")
                with jsonlines.open(data_path, mode="r") as tweets:
                    for tweet in tweets:
                        all_tweets.append(tweet)
            except BaseException:
                logger.info(f"Not on disk! scraping {users}'s tweets now")
                tweets = self.utils.user_lookup_sns(user, 10000)
                tweet_file_path = _safe_join(outdir, f'{sanitized_user}_tweets.jsonl')
                with jsonlines.open(tweet_file_path, mode='a') as writer:
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
                    cluster_labels_dir = os.path.abspath("data/cluster_labels")
                    if not os.path.exists(cluster_labels_dir):
                        os.makedirs(cluster_labels_dir, exist_ok=True)
                    sanitized_user0 = safe_filename(users[0])
                    cluster_path = _safe_join(cluster_labels_dir, f"{sanitized_user0}_id_to_cluster_label_{n_clusters}")
                    with open(cluster_path, "r") as f:
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

                    cluster_labels_dir = os.path.abspath("data/cluster_labels")
                    if not os.path.exists(cluster_labels_dir):
                        os.makedirs(cluster_labels_dir, exist_ok=True)
                    sanitized_user0 = safe_filename(users[0])
                    cluster_path = _safe_join(cluster_labels_dir, f"{sanitized_user0}_id_to_cluster_label_{n_clusters}")
                    with open(cluster_path, 'w') as f:
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