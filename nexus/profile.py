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


def _safe_join(base: str, *paths: str) -> str:
    """
    Safely join one or more path components to the base path, ensuring the final path is a subpath of base.
    Raises ValueError on invalid paths.
    """
    base_abs = os.path.abspath(base)
    final_path = os.path.abspath(os.path.join(base, *paths))
    if not final_path.startswith(base_abs + os.sep) and final_path != base_abs:
        raise ValueError(f"Unsafe path detected: {final_path} (base: {base_abs})")
    return final_path

def _is_safe_filename(filename: str) -> bool:
    """
    Checks if the filename does not contain dangerous sequences.
    """
    if os.path.isabs(filename):
        return False
    if '..' in filename.split(os.sep):
        return False
    if os.sep in filename or (os.altsep and os.altsep in filename):
        return False
    return True

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
        # Secure outdir
        base_data_dir = "data"
        if not outdir:
            outdir = base_data_dir
        outdir_abs = os.path.abspath(outdir)
        allowed_base_abs = os.path.abspath(base_data_dir)
        if not (outdir_abs.startswith(allowed_base_abs + os.sep) or outdir_abs == allowed_base_abs):
            raise ValueError(f"Unsafe output directory: {outdir}")

        os.makedirs(outdir_abs, exist_ok=True)
        lookup_amount = 10000
        for user in users:
            # Ensure safe user-specific file name
            safe_user = user.replace("/", "_")
            if not _is_safe_filename(safe_user + "_tweets.jsonl"):
                raise ValueError(f"Unsafe user/file name: {safe_user}")

            tweets = [{"text": p.clean(tweet["full_text"]), "created_at": tweet["created_at"]} for tweet in
                      self.utils.user_lookup(user, lookup_amount)]
            tweet_path = os.path.join(outdir_abs, f"{safe_user}_tweets.jsonl")
            with jsonlines.open(tweet_path, mode='a') as writer:
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

        # Sanitize outdir
        base_data_dir = "data"
        if not outdir:
            outdir = base_data_dir
        outdir_abs = os.path.abspath(outdir)
        allowed_base_abs = os.path.abspath(base_data_dir)
        if not (outdir_abs.startswith(allowed_base_abs + os.sep) or outdir_abs == allowed_base_abs):
            raise ValueError(f"Unsafe output directory: {outdir}")

        os.makedirs(outdir_abs, exist_ok=True)

        for user in tqdm(users):
            try:
                logger.info(f"Loading {user}'s tweets from disk")
                safe_user = user.replace("/", "_")
                if not _is_safe_filename(safe_user + "_tweets.jsonl"):
                    raise ValueError(f"Unsafe user/file name: {safe_user}")
                data_path = os.path.join(outdir_abs, f"{safe_user}_tweets.jsonl")
                # Ensure data_path is really inside the outdir_abs
                data_path = _safe_join(outdir_abs, f"{safe_user}_tweets.jsonl")
                with jsonlines.open(data_path, mode="r") as tweets:
                    for tweet in tweets:
                        all_tweets.append(tweet)
            except BaseException:
                logger.info(f"Not on disk! scraping {users}'s tweets now")
                tweets = self.utils.user_lookup_sns(user, 10000)
                safe_user = user.replace("/", "_")
                if not _is_safe_filename(safe_user + "_tweets.jsonl"):
                    raise ValueError(f"Unsafe user/file name: {safe_user}")
                tweet_path = os.path.join(outdir_abs, f"{safe_user}_tweets.jsonl")
                # Ensure tweet_path is inside the outdir_abs
                tweet_path = _safe_join(outdir_abs, f"{safe_user}_tweets.jsonl")
                with jsonlines.open(tweet_path, mode='a') as writer:
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
            cluster_labels_base = os.path.join("data", "cluster_labels")
            os.makedirs(cluster_labels_base, exist_ok=True)
            for n_clusters in n_cluster_docs:
                logger.info(f"computing {n_clusters} cluster layer")
                # Compose the filename and validate
                safe_user = users[0].replace("/", "_")
                cluster_file = f"{safe_user}_id_to_cluster_label_{n_clusters}"
                if not _is_safe_filename(cluster_file):
                    raise ValueError(f"Unsafe cluster label filename: {cluster_file}")
                cluster_file_path = _safe_join(cluster_labels_base, cluster_file)
                try:
                    with open(cluster_file_path, "r") as f:
                        id_to_cluster_label = json.load(f)
                    logger.info("Loaded all resources from disk")
                    print(id_to_cluster_label[-1])
                except BaseException:
                    # remake clusters
                    id_to_cluster_label = {}

                    # Embedding path handling
                    embeddings_dir = "embeddings"
                    os.makedirs(embeddings_dir, exist_ok=True)
                    # Determine a safe embedding_path
                    embedding_path_candidate = embedding_path if embedding_path else f"{safe_user}.npy"
                    embedding_file = os.path.basename(embedding_path_candidate)
                    if not _is_safe_filename(embedding_file):
                        raise ValueError(f"Unsafe embedding file name: {embedding_file}")

                    # Always store/load from 'embeddings' directory
                    resolved_embedding_path = _safe_join(embeddings_dir, embedding_file)

                    try:
                        logger.info("Loading embeddings from disk.")
                        embeddings = np.load(resolved_embedding_path, allow_pickle=False)
                    except BaseException:
                        logger.info("Embedding with Cohere")
                        cohere_api_key = os.getenv("COHERE_KEY")
                        embedder = CohereEmbedder(cohere_api_key=cohere_api_key)
                        embeddings = np.array(embedder.embed(texts=[datum['full_text'] for datum in all_tweets])).squeeze()
                        with open(resolved_embedding_path, 'wb') as f:
                            np.save(f, embeddings)
                    logger.info("Running Kmeans to generate clusters")
                    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
                    for datum, cluster_id in zip(all_tweets, [int(i) for i in list(kmeans.labels_)]):
                        id_to_cluster_label[datum['id']] = cluster_id

                    with open(cluster_file_path, 'w') as f:
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