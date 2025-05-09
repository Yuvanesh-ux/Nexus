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
import unicodedata

load_dotenv()


class Profile:
    def __init__(self):
        self.utils = Utils()
        self.atlas = AtlasClient()

    def _sanitize_filename(self, username: str) -> str:
        """
        Sanitizes the input username for safe use in filenames:
         - Removes any slashes, backslashes, or dot-dot patterns
         - Removes illegal/suspicious characters
         - Only allows alphanumeric, underscore, hyphen, and dot
         - Optionally trims length if needed
        """
        # Remove directory traversal patterns
        username = re.sub(r'(\.\.[/\\])+', '', username)
        # Remove all slashes and backslashes
        username = username.replace('/', '').replace('\\', '')
        # Remove all but allowed characters
        username = re.sub(r'[^A-Za-z0-9._-]', '', username)
        # Optionally, trim overly long usernames for safety
        return username[:128]

    @staticmethod
    def _is_valid_tweet_content(text: str) -> bool:
        """
        Checks if the tweet text is valid for ingestion into ML pipeline.
        - Checks for unicode control chars and invisibles
        - Length thresholding (not too short/long)
        - Detects repeated substrings (simple backdoor attempts)
        - Checks for abnormal non-latin characters (rudimentary language check)
        - Blocks strings with excessive links, mentions, hashtags
        """
        if not isinstance(text, str):
            return False
        # Remove whitespace for effective length
        content = text.strip()
        # Length between 30 and 400 (Twitter hard max is 280, maybe longer for threads)
        if len(content) < 30 or len(content) > 400:
            return False
        # Block excessive URLs (>2)
        url_count = len(re.findall(r'https?://', content))
        if url_count > 2:
            return False
        # Block excessive mentions (>5)
        mention_count = len(re.findall(r'@\w+', content))
        if mention_count > 5:
            return False
        # Block excessive hashtags (>7)
        hashtag_count = len(re.findall(r'#\w+', content))
        if hashtag_count > 7:
            return False
        # Block control/unprintable chars
        if any(unicodedata.category(c)[0] == "C" and c not in ("\n", "\r", "\t") for c in content):
            return False
        # Check for non-latin (heuristic; could be improved)
        nonlatin = re.sub(r"[a-zA-Z0-9 .,;:?!'\"()\-\[\]{}@#/]", '', content)
        # If >30% of text is non-latin non-punctuation, suspicious
        if len(nonlatin) > 0.3 * len(content):
            return False
        # Block repeated n-grams (simple backdoor triggers, e.g. 'xyz xyz xyz ...')
        repeated_word_seq = re.search(r'(\b\w+\b)( \1\b)+', content)
        if repeated_word_seq is not None:
            return False
        # Block extremely high character repetition (e.g. 'A' * 50)
        for c in set(content):
            if content.count(c) > 40 and c != ' ':
                return False
        return True

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
            safe_user = self._sanitize_filename(user)
            user_file_path = os.path.join(outdir, f'{safe_user}_tweets.jsonl')
            os.makedirs(os.path.dirname(user_file_path), exist_ok=True)
            with jsonlines.open(user_file_path, mode='a') as writer:
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
                safe_user = self._sanitize_filename(user)
                data_path = os.path.join(outdir, f"{safe_user}_tweets.jsonl")
                with jsonlines.open(data_path, mode="r") as tweets:
                    for tweet in tweets:
                        # VALIDATION: Filter for valid tweets only
                        text_check = tweet.get("full_text", tweet.get("text", ""))
                        if Profile._is_valid_tweet_content(str(text_check)):
                            all_tweets.append(tweet)
                        else:
                            logger.warning(f"Filtered suspicious tweet from disk for user {user}: {text_check!r}")
            except BaseException:
                logger.info(f"Not on disk! scraping {users}'s tweets now")
                tweets = self.utils.user_lookup_sns(user, 10000)
                safe_user = self._sanitize_filename(user)
                user_file_path = os.path.join(outdir, f'{safe_user}_tweets.jsonl')
                os.makedirs(os.path.dirname(user_file_path), exist_ok=True)
                with jsonlines.open(user_file_path, mode='a') as writer:
                    for idx, tweet in enumerate(tweets):
                        tweet["full_text"] = p.clean(tweet["full_text"])
                        if Profile._is_valid_tweet_content(tweet["full_text"]):
                            tweet["created_at"] = str(tweet["created_at"])
                            all_tweets.append(tweet)
                            writer.write(tweet)
                        else:
                            logger.warning(f"Filtered suspicious scraped tweet for user {user}: {tweet['full_text']!r}")


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