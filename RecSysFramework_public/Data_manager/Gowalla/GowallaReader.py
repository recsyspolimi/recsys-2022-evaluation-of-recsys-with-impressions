#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25/04/2019

@author: Anonymous
"""



import gzip, os
import pandas as pd
from Data_manager.DatasetMapperManager import DatasetMapperManager
from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import download_from_URL, remove_Dataframe_duplicates


class GowallaReader(DataReader):

    DATASET_URL = "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz"
    DATASET_SUBFOLDER = "Gowalla/"
    AVAILABLE_URM = ["URM_all", "URM_occurrence"]


    ZIP_NAME = "loc-gowalla_totalCheckins.txt.gz"
    FILE_RATINGS_PATH =   "loc-gowalla_totalCheckins.txt"


    AVAILABLE_ICM = []
    DATASET_SPECIFIC_MAPPER = []

    IS_IMPLICIT = False


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        self._print("Loading original data")

        folder_path =  self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            compressed_file = gzip.open(folder_path + self.ZIP_NAME, )

        except FileNotFoundError:

            self._print("Unable to find data zip file. Downloading...")
            download_from_URL(self.DATASET_URL, folder_path, self.ZIP_NAME)

            compressed_file = gzip.open(folder_path + self.ZIP_NAME)


        URM_path = folder_path + self.FILE_RATINGS_PATH

        decompressed_file = open(URM_path, "w")

        self._save_GZ_in_text_file(compressed_file, decompressed_file)

        decompressed_file.close()

        self._print("Loading Interactions")
        URM_all_dataframe = pd.read_csv(filepath_or_buffer=URM_path, sep="\t", header=None,
                                        dtype={0:str, 4:str}, usecols=[0, 4])

        URM_all_dataframe.columns = ["UserID", "ItemID"]
        URM_all_dataframe["Data"] = 1

        URM_occurrence_dataframe = URM_all_dataframe.groupby(["UserID","ItemID"],as_index=False)["Data"].sum()

        URM_all_dataframe = remove_Dataframe_duplicates(URM_all_dataframe,
                                                        unique_values_in_columns = ['UserID', 'ItemID'],
                                                        keep_highest_value_in_col = "Data")

        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_all_dataframe, "URM_all")
        dataset_manager.add_URM(URM_occurrence_dataframe, "URM_occurrence")

        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)



        self._print("Cleaning Temporary Files")

        os.remove(URM_path)

        self._print("Loading Complete")

        return loaded_dataset





    def _save_GZ_in_text_file(self, compressed_file, decompressed_file):

        print("GowallaReader: decompressing file...")

        for line in compressed_file:
            decompressed_file.write(line.decode("utf-8"))

        decompressed_file.flush()

        print("GowallaReader: decompressing file... done!")

