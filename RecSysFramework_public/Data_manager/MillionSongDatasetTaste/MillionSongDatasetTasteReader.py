#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Anonymous
"""


import zipfile, shutil
import pandas as pd
from Data_manager.DatasetMapperManager import DatasetMapperManager
from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import download_from_URL


class MillionSongDatasetTasteReader(DataReader):
    """
    This dataset is the Taste Profile subset of the Million Song Dataset.
    http://millionsongdataset.com/tasteprofile/
    """

    DATASET_URL = "http://millionsongdataset.com/sites/default/files/challenge/train_triplets.txt.zip"
    DATASET_SUBFOLDER = "MillionSongDatasetTaste/"
    AVAILABLE_URM = ["URM_all", "URM_occurrence"]
    AVAILABLE_ICM = []

    IS_IMPLICIT = True

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER


    def _load_from_original_file(self):
        # Load data from original

        zipFile_path =  self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "train_triplets.txt.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            self._print("Unable to find data zip file. Downloading...")

            download_from_URL(self.DATASET_URL, zipFile_path, "train_triplets.txt.zip")

            dataFile = zipfile.ZipFile(zipFile_path + "train_triplets.txt.zip")


        URM_path = dataFile.extract("train_triplets.txt", path=zipFile_path + "decompressed/")

        self._print("Loading Interactions")
        URM_occurrence_dataframe = pd.read_csv(filepath_or_buffer=URM_path, sep='\t', header=None, dtype={0:str, 1:str, 2:int})
        URM_occurrence_dataframe.columns = ["UserID", "ItemID", "Data"]

        URM_all_dataframe = URM_occurrence_dataframe.copy()
        URM_all_dataframe["Data"] = 1

        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_all_dataframe, "URM_all")
        dataset_manager.add_URM(URM_occurrence_dataframe, "URM_occurrence")

        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)

        self._print("Cleaning Temporary Files")

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        self._print("Loading Complete")

        return loaded_dataset

