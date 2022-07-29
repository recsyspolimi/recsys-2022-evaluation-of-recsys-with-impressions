#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 02/06/2021

@author: Anonymous
"""



import zipfile, shutil
import pandas as pd
from Data_manager.DatasetMapperManager import DatasetMapperManager
from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import download_from_URL
import scipy.sparse as sps

class JesterJokesReader(DataReader):

    DATASET_URL = "http://eigentaste.berkeley.edu/dataset/"
    DATASET_SUBFOLDER = "JesterJokes/"
    AVAILABLE_URM = ["URM_all"]
    AVAILABLE_ICM = []
    AVAILABLE_UCM = []
    DATASET_SPECIFIC_MAPPER = []


    def __init__(self):
        super(JesterJokesReader, self).__init__()


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        self._print("Loading original data")

        zipFile_path = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER
        dataFile_list = [None]*3

        try:
            for file_index in range(3):
                file_name = "jester_dataset_1_{}.zip".format(file_index+1)
                dataFile_list[file_index] = zipfile.ZipFile(zipFile_path + file_name)

        except (FileNotFoundError, zipfile.BadZipFile):

            self._print("Unable to find data zip file. Downloading...")

            for file_index in range(3):
                file_name = "jester_dataset_1_{}.zip".format(file_index+1)
                download_from_URL(self.DATASET_URL + file_name, zipFile_path, file_name)
                dataFile_list[file_index] = zipfile.ZipFile(zipFile_path + file_name)

        # Format:
        # Data files are in .zip format, when unzipped, they are in Excel (.xls) format
        # Ratings are real values ranging from -10.00 to +10.00 (the value "99" corresponds to "null" = "not rated").
        # One row per user
        # The first column gives the number of jokes rated by that user. The next 100 columns give the ratings for jokes 01 - 100.
        # The sub-matrix including only columns {5, 7, 8, 13, 15, 16, 17, 18, 19, 20} is dense. Almost all users have rated those jokes (see discussion of "universal queries" in the above paper).

        dataframe_list = [None]*3

        for file_index in range(3):
            URM_part_path = dataFile_list[file_index].extract("jester-data-{}.xls".format(file_index+1), path=zipFile_path + "decompressed/")
            dataframe = pd.read_excel(URM_part_path, sheet_name="jester-data-{}-new".format(file_index+1), header=None, index_col=None)

            # Remove column containing the number of rated jokes per user
            dataframe.drop([0], axis=1, inplace=True)

            # Replacing negatively rated and non-rated jokes as 0
            dataframe.replace(99,0, inplace=True)
            dataframe.clip(0, inplace=True)
            dataframe_list[file_index] = dataframe

        dataframe_all = pd.concat(dataframe_list, ignore_index = True)
        URM_all = sps.coo_matrix(dataframe_all.values)
        URM_all.eliminate_zeros()

        dataframe_all = pd.DataFrame({"UserID": URM_all.row,
                                      "ItemID": URM_all.col,
                                      "Data": [1]*len(URM_all.row)})

        dataframe_all["UserID"] = dataframe_all["UserID"].apply(str)
        dataframe_all["ItemID"] = dataframe_all["ItemID"].apply(str)

        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(dataframe_all, "URM_all")

        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)

        self._print("Cleaning Temporary Files")

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        self._print("Loading Complete")

        return loaded_dataset



