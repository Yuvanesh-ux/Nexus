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
import posixpath  # For safe path manipulation
from dotenv import load_dotenv

load_dotenv()


def sanitize_filename(filename):
    """Sanitize a filename to prevent path traversal."""
    # Remove any directory traversal components
    safe_name = posixpath.basename(filename)
    # Remove any other potentially dangerous characters
    safe_name = re.sub(r'[^\w\.-]', '_', safe_name)
    return safe_name


def safe_path_join(base_dir, *paths):
    """Safely join paths to prevent path traversal."""
    # Make sure base_dir is absolute
    base_dir = os.path.abspath(base_dir)
    # Sanitize and join paths
    safe_paths = [sanitize_filename(p) for p in paths]
    path = os.path.join(base_dir, *safe_paths)
    # Ensure result is still within base_dir
    if not os.path.abspath(path).startswith(base_dir):
        raise ValueError(f"Path traversal attempt detected: {path}")
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
        for user in users:
            tweets = [{"text": p.clean(tweet["full_text"]), "created_at": tweet["created_at"]} for tweet in
                      self.utils.user_lookup(user, lookup_amount)]
            safe_filename = f"{sanitize_filename(user)}_tweets.jsonl"
            safe_path = safe_path_join(outdir, safe_filename)
            with jsonlines.open(safe_path, mode='a') as writer:
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


        for user in tqdm(users):
            try:
                logger.info(f"Loading {user}'s tweets from disk")
                safe_filename = f"{sanitize_filename(user)}_tweets.jsonl"
                data_path = safe_path_join(outdir, safe_filename)
                with jsonlines.open(data_path, mode="r") as tweets:
                    for tweet in tweets:
                        all_tweets.append(tweet)
            except BaseException:
                logger.info(f"Not on disk! scraping {users}'s tweets now")
                tweets = self.utils.user_lookup_sns(user, 10000)
                safe_filename = f"{sanitize_filename(user)}_tweets.jsonl"
                safe_path = safe_path_join(outdir, safe_filename)
                with jsonlines.open(safe_path, mode='a') as writer:
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
                    safe_user = sanitize_filename(users[0])
                    cluster_path = f"data/cluster_labels/{safe_user}_id_to_cluster_label_{n_clusters}"
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

                    safe_user = sanitize_filename(users[0])
                    cluster_path = f"data/cluster_labels/{safe_user}_id_to_cluster_label_{n_clusters}"
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