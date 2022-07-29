#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Anonymous
"""

import pandas as pd
import zipfile, shutil
from Data_manager.DataReader import DataReader
from Data_manager.DatasetMapperManager import DatasetMapperManager
from Data_manager.DataReader_utils import download_from_URL, remove_Dataframe_duplicates




class DeliciousHetrec2011Reader(DataReader):

    DATASET_URL = "http://files.grouplens.org/datasets/hetrec2011/hetrec2011-delicious-2k.zip"
    DATASET_SUBFOLDER = "DeliciousHetrec2011/"
    AVAILABLE_URM = ["URM_all", "URM_timestamp"]
    AVAILABLE_ICM = ['ICM_tags']
    AVAILABLE_UCM = ['UCM_social']

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        self._print("Loading original data")

        folder_path = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER


        try:

            dataFile = zipfile.ZipFile(folder_path + "hetrec2011-delicious-2k.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            self._print("Unable to find or extract data zip file. Downloading...")

            download_from_URL(self.DATASET_URL, folder_path, "hetrec2011-delicious-2k.zip")

            dataFile = zipfile.ZipFile(folder_path + "hetrec2011-delicious-2k.zip")



        URM_path = dataFile.extract("user_taggedbookmarks-timestamps.dat", path=folder_path + "decompressed")
        ICM_path = dataFile.extract("bookmark_tags.dat", path=folder_path + "decompressed")
        UCM_path = dataFile.extract("user_contacts-timestamps.dat", path=folder_path + "decompressed")

        self._print("Loading Interactions")
        URM_all_dataframe = pd.read_csv(filepath_or_buffer=URM_path, sep="\t",
                                         header= 0, dtype={0:str, 1:str, 2:str, 3:float})
        URM_all_dataframe.columns = ["UserID", "ItemID", "tagID", "timestamp"]

        URM_all_dataframe = remove_Dataframe_duplicates(URM_all_dataframe, unique_values_in_columns = ['UserID', 'ItemID'],
                                                        keep_highest_value_in_col = "timestamp")


        self._print('Loading Item Tags')
        ICM_tags_dataframe = pd.read_csv(filepath_or_buffer=ICM_path, sep="\t",
                                         header= 0, dtype={0:str, 1:str, 2:float})

        self._print('Loading User Contacts')
        UCM_social_dataframe = pd.read_csv(filepath_or_buffer=UCM_path, sep="\t",
                                         header= 0, dtype={0:str, 1:str, 2:float})

        URM_timestamp_dataframe = URM_all_dataframe.copy().drop(columns=["tagID"])
        URM_all_dataframe = URM_all_dataframe.drop(columns=["timestamp"])

        URM_timestamp_dataframe.columns = ["UserID", "ItemID", "Data"]
        URM_all_dataframe.columns = ["UserID", "ItemID", "Data"]
        ICM_tags_dataframe.columns = ["ItemID", "FeatureID", "Data"]
        UCM_social_dataframe.columns = ["UserID", "FeatureID", "Data"]

        URM_all_dataframe['Data'] = 1
        ICM_tags_dataframe['Data'] = 1
        UCM_social_dataframe['Data'] = 1

        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_all_dataframe, "URM_all")
        dataset_manager.add_URM(URM_timestamp_dataframe, "URM_timestamp")
        dataset_manager.add_ICM(ICM_tags_dataframe, "ICM_tags")
        dataset_manager.add_UCM(UCM_social_dataframe, "UCM_social")

        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)


        self._print("Cleaning Temporary Files")

        shutil.rmtree(folder_path + "decompressed", ignore_errors=True)

        self._print("Loading Complete")

        return loaded_dataset

