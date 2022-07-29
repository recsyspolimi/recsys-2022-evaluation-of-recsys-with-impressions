#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Anonymous
"""


import numpy as np
import pandas as pd
import zipfile, os, shutil
from Data_manager.DatasetMapperManager import DatasetMapperManager
from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import remove_Dataframe_duplicates
import gc

class TwitterChallenge2020Reader(DataReader):

    DATASET_URL = "https://polimi365-my.sharepoint.com/:f:/g/personal/10322330_polimi_it/Eiuw9hVjthVDjg0KQD7AZzIB2tTHlMS-qhprH14TFVfwiA?e=nXNnCo"
    DATASET_SUBFOLDER = "TwitterChallenge2020/"

    DATASET_SPECIFIC_MAPPER = []
    # AVAILABLE_ICM = ["ICM_text_tokens", "ICM_text_hashtags", "ICM_media"]
    AVAILABLE_ICM = ["ICM_text_tokens", "ICM_text_hashtags", "ICM_media"]
    AVAILABLE_URM = ["URM_all"]

    def __init__(self, min_occurrences_to_select_tweet = 3, rows_to_load = None,
                 training_file_path = None):
        super(TwitterChallenge2020Reader, self).__init__()

        self.min_occurrences_to_select_tweet = min_occurrences_to_select_tweet
        self.rows_to_load = rows_to_load

        assert training_file_path is not None
        self.training_file_path = training_file_path


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        self._print("Loading original data")

        compressed_file_folder = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER
        decompressed_file_folder = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER


        try:

            # dataFile = zipfile.ZipFile(compressed_file_folder + "RecSys Challenge 2019 Trivago.zip")
            # URM_path = dataFile.extract("train.csv", path=decompressed_file_folder + "decompressed/")
            # ICM_path = dataFile.extract("item_metadata.csv", path=decompressed_file_folder + "decompressed/")
            pass

        except (FileNotFoundError, zipfile.BadZipFile):

            self._print("Unable to find data zip file.")
            self._print("Automatic download not available, please ensure the compressed data file is in folder {}.".format(compressed_file_folder))
            self._print("Data can be downloaded here: {}".format(self.DATASET_URL))

            # If directory does not exist, create
            if not os.path.exists(compressed_file_folder):
                os.makedirs(compressed_file_folder)

            raise FileNotFoundError("Automatic download not available.")


        # The user creating the tweet is the ENGAGING, the users interacting with hit are the ENGAGED
        all_features = ["text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",
                        "tweet_type","language", "tweet_timestamp",
                        "engaged_with_user_id", "engaged_with_user_follower_count",
                        "engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",
                        "enaging_user_id", "enaging_user_follower_count", "enaging_user_following_count", "enaging_user_is_verified",
                        "enaging_user_account_creation", "engagee_follows_engager"]

        all_labels = ["reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"]
        all_columns = [*all_features, *all_labels]

        all_columns_to_dtype = {
            "text_tokens":str,
            "hashtags":str,
            "tweet_id":"category",
            "present_media":str,
            "present_links":str,
            "present_domains":str,
            "tweet_type":str,
            "language":str,
            "tweet_timestamp":float,
            "engaged_with_user_id":str,
            "engaged_with_user_follower_count":str,
            "engaged_with_user_following_count":str,
            "engaged_with_user_is_verified":str,
            "engaged_with_user_account_creation":str,
            "enaging_user_id":"category",
            "enaging_user_follower_count":int,
            "enaging_user_following_count":int,
            "enaging_user_is_verified":str,
            "enaging_user_account_creation":str,
            "engagee_follows_engager":str,
            "reply_timestamp":float,
            "retweet_timestamp":float,
            "retweet_with_comment_timestamp":float,
            "like_timestamp":float,
        }

        all_columns_to_idx = dict(zip(all_columns, range(len(all_columns))))

        self._print("Loading Interactions")

        # columns_to_read = ["text_tokens", "hashtags", "tweet_id", "present_media", "enaging_user_id",  *all_labels]
        columns_to_read = ["hashtags", "tweet_id", "present_media", "enaging_user_id", *all_labels]
        columns_to_read_idx = [all_columns_to_idx[col_label] for col_label in columns_to_read]
        columns_to_read_dtype = [all_columns_to_dtype[col_label] for col_label in columns_to_read]

        original_data_dataframe = pd.read_csv(filepath_or_buffer=self.training_file_path, encoding="utf-8",
                                            sep="\x01", header=None, nrows=self.rows_to_load,
                                            usecols=columns_to_read_idx,
                                            dtype= {columns_to_read_idx[index]:columns_to_read_dtype[index] for index in range(len(columns_to_read))})

        original_data_dataframe.columns = columns_to_read

        original_data_dataframe["AnyInteraction"] = pd.notna(original_data_dataframe["reply_timestamp"].values) | \
                                                    pd.notna(original_data_dataframe["retweet_timestamp"].values) | \
                                                    pd.notna(original_data_dataframe["retweet_with_comment_timestamp"].values) | \
                                                    pd.notna(original_data_dataframe["like_timestamp"].values)

        original_data_dataframe = original_data_dataframe[original_data_dataframe["AnyInteraction"]]

        tweet_id_occurrence = original_data_dataframe['tweet_id'].value_counts()
        tweet_id_to_select = tweet_id_occurrence[tweet_id_occurrence>self.min_occurrences_to_select_tweet]

        original_data_dataframe = original_data_dataframe[original_data_dataframe["tweet_id"].isin(tweet_id_to_select.index)]


        URM_all_dataframe = original_data_dataframe[["tweet_id", "enaging_user_id"]].copy()
        URM_all_dataframe.rename(columns={"tweet_id": "ItemID",
                                          "enaging_user_id": "UserID",
                                          }, inplace=True)
        URM_all_dataframe["Data"] = 1

        # Multiple interactions exist with the same item
        URM_all_dataframe = URM_all_dataframe.groupby(["UserID", "ItemID"], as_index=False, observed=True)["Data"].max()






        self._print("Loading Item Features")

        # Remove duplicated tweets, keep only the first occurrence
        original_data_dataframe = original_data_dataframe[~original_data_dataframe.duplicated(subset=['tweet_id'], keep='first')]
        #
        # def _split_columns_in_list(original_data_dataframe, column_label, separator):
        #     original_data_dataframe[column_label] = original_data_dataframe[column_label].str.split(separator)
        #     original_data_dataframe[column_label] = [x if type(x) is list else [] for x in original_data_dataframe[column_label]]
        #
        #
        # for colum_label in ["text_tokens", "hashtags", "present_media", "present_links", "present_domains"]:
        #     _split_columns_in_list(original_data_dataframe, colum_label, "\t")
        #
        # ICM_text_tokens_dataframe = pd.DataFrame(original_data_dataframe["text_tokens"].tolist(),
        #                                          index=original_data_dataframe["tweet_id"].tolist()).stack()
        # ICM_text_tokens_dataframe = ICM_text_tokens_dataframe.reset_index()[["level_0", 0]]
        # ICM_text_tokens_dataframe.columns = ['ItemID', 'FeatureID']
        # ICM_text_tokens_dataframe["Data"] = 1

        # ICM_text_hashtags_dataframe = pd.DataFrame(original_data_dataframe["hashtags"].tolist(),
        #                                          index=original_data_dataframe["tweet_id"].tolist()).stack()

        def _get_feature_dataframe_from_list_colum(original_data_dataframe, column_label, separator):
            data_lists = original_data_dataframe[column_label].str.split(separator).tolist()
            data_lists = [x if type(x) is list else [] for x in data_lists]

            ICM_feature_dataframe = pd.DataFrame(data_lists, index=original_data_dataframe["tweet_id"].tolist()).stack()
            ICM_feature_dataframe = ICM_feature_dataframe.reset_index()[["level_0", 0]]
            ICM_feature_dataframe.columns = ['ItemID', 'FeatureID']
            ICM_feature_dataframe["Data"] = 1

            return ICM_feature_dataframe


        # ICM_text_tokens_dataframe = _get_feature_dataframe_from_list_colum(original_data_dataframe, "text_tokens", "\t")
        ICM_text_hashtags_dataframe = _get_feature_dataframe_from_list_colum(original_data_dataframe, "hashtags", "\t")
        ICM_text_media_dataframe = _get_feature_dataframe_from_list_colum(original_data_dataframe, "present_media", "\t")








        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_all_dataframe, "URM_all")
        # dataset_manager.add_ICM(ICM_text_tokens_dataframe, "ICM_text_tokens")
        dataset_manager.add_ICM(ICM_text_hashtags_dataframe, "ICM_text_hashtags")
        dataset_manager.add_ICM(ICM_text_media_dataframe, "ICM_media")


        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)



        self._print("Cleaning Temporary Files")

        shutil.rmtree(decompressed_file_folder + "decompressed/", ignore_errors=True)

        self._print("Loading Complete")

        return loaded_dataset

