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


class Profile:
    def __init__(self):
        self.utils = Utils()
        self.atlas = AtlasClient()

    def validate_and_sanitize_tweet(self, text: str) -> str:
        """
        Performs robust validation and sanitization of tweet text.
        Returns sanitized text or empty string if validation fails.
        """
        # Basic validation
        if not text or len(text) < 10 or len(text) > 1000:
            return ""
        
        # Check for suspicious patterns
        suspicious_patterns = [
            '<script', 'javascript:', 'onerror=', 'onclick=', 
            'SELECT ', 'INSERT ', 'DELETE ', 'UPDATE ', 'DROP ', 
            '<?php', '<?=', '<%', 
            'UNION SELECT', 'OR 1=1', 'AND 1=1'
        ]
        
        lower_text = text.lower()
        for pattern in suspicious_patterns:
            if pattern.lower() in lower_text:
                return ""
        
        # Additional sanitization
        sanitized = re.sub(r'[^\w\s.,!?-]', '', text)  # Remove special characters
        sanitized = re.sub(r'(.)\1{5,}', r'\1\1\1', sanitized)  # Limit character repetition
        
        return sanitized.strip()

    def validate_embeddings(self, embeddings: np.ndarray) -> bool:
        """
        Validates embeddings to detect potential manipulation.
        """
        # Check for invalid values
        if np.isnan(embeddings).any() or np.isinf(embeddings).any():
            return False
        
        # Check statistical properties
        mean = np.mean(embeddings)
        std = np.std(embeddings)
        
        # Define acceptable ranges
        if abs(mean) > 5 or std < 0.01 or std > 5:
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
                data_path = os.path.join(outdir, f"{user}_tweets.jsonl")
                with jsonlines.open(data_path, mode="r") as tweets:
                    for tweet in tweets:
                        all_tweets.append(tweet)
            except BaseException:
                logger.info(f"Not on disk! scraping {users}'s tweets now")
                tweets = self.utils.user_lookup_sns(user, 10000)
                with jsonlines.open(f'{outdir}/{user}_tweets.jsonl', mode='a') as writer:
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
                        
                        # Validate and sanitize tweets before embedding
                        sanitized_tweets = []
                        valid_indices = []
                        for idx, datum in enumerate(all_tweets):
                            sanitized = self.validate_and_sanitize_tweet(datum['full_text'])
                            if sanitized:  # If validation passed
                                sanitized_tweets.append(sanitized)
                                valid_indices.append(idx)

                        logger.info(f"Validated {len(sanitized_tweets)} out of {len(all_tweets)} tweets")

                        if len(sanitized_tweets) < 10:  # Ensure we have enough valid tweets
                            logger.error("Too few valid tweets for reliable embedding")
                            raise ValueError("Insufficient valid tweets for embedding")

                        # Embed only the valid tweets
                        raw_embeddings = np.array(embedder.embed(texts=sanitized_tweets)).squeeze()

                        # Validate the embeddings
                        if not self.validate_embeddings(raw_embeddings):
                            logger.error("Embedding validation failed. Potential manipulation detected.")
                            raise ValueError("Embedding validation failed")

                        # Map embeddings back to the original tweets
                        # (invalid tweets get zero embeddings)
                        if len(raw_embeddings.shape) == 1:  # Handle single embedding case
                            embedding_dim = raw_embeddings.shape[0]
                            embeddings = np.zeros((len(all_tweets), embedding_dim))
                            embeddings[valid_indices[0]] = raw_embeddings
                        else:
                            embedding_dim = raw_embeddings.shape[1]
                            embeddings = np.zeros((len(all_tweets), embedding_dim))
                            for i, idx in enumerate(valid_indices):
                                embeddings[idx] = raw_embeddings[i]

                        logger.info(f"Successfully embedded {len(sanitized_tweets)} validated tweets")
                        with open(embedding_path, 'wb') as f:
                            np.save(f, embeddings)
                    
                    # Identify tweets with valid embeddings (non-zero)
                    valid_embedding_mask = np.sum(embeddings, axis=1) != 0
                    valid_embeddings = embeddings[valid_embedding_mask]
                    valid_indices = np.where(valid_embedding_mask)[0]
                    
                    if len(valid_embeddings) < 10:
                        logger.error("Too few valid embeddings for clustering")
                        raise ValueError("Insufficient valid embeddings for clustering")
                    
                    logger.info("Running Kmeans to generate clusters")
                    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(valid_embeddings)
                    
                    # Map cluster assignments back to all tweets
                    for i, idx in enumerate(valid_indices):
                        tweet_id = all_tweets[idx]['id']
                        id_to_cluster_label[tweet_id] = int(kmeans.labels_[i])
                    
                    # Assign a default cluster (-1) for tweets without valid embeddings
                    for datum in all_tweets:
                        if datum['id'] not in id_to_cluster_label:
                            id_to_cluster_label[datum['id']] = -1

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