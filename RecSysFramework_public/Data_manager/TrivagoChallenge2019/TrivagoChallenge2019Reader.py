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


class TrivagoChallenge2019Reader(DataReader):
    """
    Dataset of the 2019 RecSys Challenge.
    Related paper: https://dl.acm.org/doi/10.1145/3412379
    Data sources: https://recsys2019data.trivago.com/
    """

    DATASET_URL = "https://polimi365-my.sharepoint.com/:u:/g/personal/10322330_polimi_it/EcQWVFLvQF9AsPKLPeqYxF0BKDfZBWlEam_QPkbrp7zInQ?e=ZnbjYw"
    DATASET_SUBFOLDER = "TrivagoChallenge2019/"

    DATASET_SPECIFIC_MAPPER = []
    AVAILABLE_ICM = ["ICM"]
    AVAILABLE_URM = ["URM_all", "URM_position", "URM_clickout", "URM_interaction"]


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        self._print("Loading original data")

        compressed_file_folder = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER
        decompressed_file_folder = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER


        try:

            dataFile = zipfile.ZipFile(compressed_file_folder + "RecSys Challenge 2019 Trivago.zip")
            URM_path = dataFile.extract("train.csv", path=decompressed_file_folder + "decompressed/")
            ICM_path = dataFile.extract("item_metadata.csv", path=decompressed_file_folder + "decompressed/")

        except (FileNotFoundError, zipfile.BadZipFile):

            self._print("Unable to find data zip file.")
            self._print("Automatic download not available, please ensure the compressed data file is in folder {}.".format(compressed_file_folder))
            self._print("Data can be downloaded here: {}".format(self.DATASET_URL))

            # If directory does not exist, create
            if not os.path.exists(compressed_file_folder):
                os.makedirs(compressed_file_folder)

            raise FileNotFoundError("Automatic download not available.")


        self._print("Loading Item Features")
        ICM_dataframe = pd.read_csv(filepath_or_buffer=ICM_path, sep=",", header=0, dtype={0:str, 1:str})
        ICM_dataframe.columns = ["ItemID", "TagList"]

        # Split TagList in order to obtain a dataframe with a tag per row
        ICM_dataframe = pd.DataFrame(ICM_dataframe["TagList"].str.split('|').tolist(), index=ICM_dataframe["ItemID"]).stack()
        ICM_dataframe = ICM_dataframe.reset_index()[[0, 'ItemID']]
        ICM_dataframe.columns = ['FeatureID', 'ItemID']
        ICM_dataframe["Data"] = 1

        self._print("Loading Interactions")
        # user_id,session_id,timestamp,step,action_type,reference,platform,city,device,current_filters,impressions,prices
        dtype_cols= dict((index,int) if index==2 else (index,str) for index in range(12))
        original_data_dataframe = pd.read_csv(filepath_or_buffer=URM_path, sep=",", header=0, dtype=dtype_cols)

        # Select only clickouts
        URM_clickout_dataframe = original_data_dataframe[original_data_dataframe["action_type"] == "clickout item"]
        URM_clickout_dataframe = URM_clickout_dataframe[["user_id", "reference", "timestamp"]]
        URM_clickout_dataframe.columns = ["UserID", "ItemID", "Timestamp"]
        URM_clickout_dataframe["Interaction"] = 1

        URM_clickout_dataframe = remove_Dataframe_duplicates(URM_clickout_dataframe,
                                                        unique_values_in_columns = ['UserID', 'ItemID'],
                                                        keep_highest_value_in_col = "Timestamp")


        URM_interaction_dataframe = original_data_dataframe[original_data_dataframe["action_type"].str.contains("interaction item")]
        URM_interaction_dataframe = URM_interaction_dataframe[["user_id", "reference", "timestamp"]]
        URM_interaction_dataframe.columns = ["UserID", "ItemID", "Timestamp"]
        URM_interaction_dataframe["Interaction"] = 1

        URM_interaction_dataframe = remove_Dataframe_duplicates(URM_interaction_dataframe,
                                                                unique_values_in_columns = ['UserID', 'ItemID'],
                                                                keep_highest_value_in_col = "Timestamp")


        # URM_timestamp_dataframe = URM_clickout_dataframe.copy().drop(columns=["Interaction"])
        URM_clickout_dataframe = URM_clickout_dataframe.drop(columns=["Timestamp"])
        URM_interaction_dataframe = URM_interaction_dataframe.drop(columns=["Timestamp"])
        # URM_timestamp_dataframe.columns = ["UserID", "ItemID", "Data"]
        URM_clickout_dataframe.columns = ["UserID", "ItemID", "Data"]
        URM_interaction_dataframe.columns = ["UserID", "ItemID", "Data"]

        URM_all_dataframe = URM_clickout_dataframe.copy()
        # URM_all_dataframe["Data"] = 10
        URM_all_dataframe = pd.concat([URM_all_dataframe, URM_interaction_dataframe])

        URM_all_dataframe = remove_Dataframe_duplicates(URM_all_dataframe,
                                                        unique_values_in_columns = ['UserID', 'ItemID'],
                                                        keep_highest_value_in_col = "Data")

        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_all_dataframe, "URM_all")
        dataset_manager.add_URM(URM_clickout_dataframe, "URM_clickout")
        dataset_manager.add_URM(URM_interaction_dataframe, "URM_interaction")
        # dataset_manager.add_URM(URM_timestamp_dataframe, "URM_timestamp")
        dataset_manager.add_ICM(ICM_dataframe, "ICM")

        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)

        self._print("Cleaning Temporary Files")

        shutil.rmtree(decompressed_file_folder + "decompressed/", ignore_errors=True)

        self._print("Loading Complete")

        return loaded_dataset

