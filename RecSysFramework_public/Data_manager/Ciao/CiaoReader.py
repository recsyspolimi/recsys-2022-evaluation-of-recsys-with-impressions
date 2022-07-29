#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Anonymous
"""

import os, zipfile, h5py, shutil
import scipy.io
import pandas as pd
from Data_manager.DatasetMapperManager import DatasetMapperManager
from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import download_from_URL

def _read_categories_dict(file_path):

    f_data = open(file_path, "r")
    categories_dict = {}

    for row in f_data:
        row = row.strip("\n")
        ID, name = row.split(' ', 1)
        categories_dict[int(ID)] = name

    return categories_dict


class CiaoReader(DataReader):

    DATASET_URL = "https://www.cse.msu.edu/~tangjili/datasetcode/ciao.zip"
    DATASET_SUBFOLDER = "Ciao/"
    AVAILABLE_ICM = ["ICM_categories"]
    DATASET_SPECIFIC_MAPPER = []

    IS_IMPLICIT = False


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        self._print("Loading original data")

        zipFile_path = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "ciao.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            self._print("Unable to find data zip file. Downloading...")

            download_from_URL(self.DATASET_URL, zipFile_path, "ciao.zip")
            download_from_URL("https://www.cse.msu.edu/~tangjili/datasetcode/catalog_ciao.txt", zipFile_path, "catalog_ciao.txt")

            dataFile = zipfile.ZipFile(zipFile_path + "ciao.zip")


        self._print("Loading Interactions")
        URM_all_path = dataFile.extract("ciao/rating.mat", path=zipFile_path + "decompressed/")

        f = scipy.io.loadmat(URM_all_path)

        dataframe = pd.DataFrame(data=f["rating"], columns=["UserID", "ItemID", "CategoryID", "Rating", "Helpfulness"])
        dataframe["UserID"] = dataframe["UserID"].astype(str)
        dataframe["ItemID"] = dataframe["ItemID"].astype(str)

        URM_all_dataframe = dataframe[["UserID", "ItemID","Rating"]].copy()
        URM_all_dataframe.columns = ["UserID", "ItemID", "Data"]
        URM_all_dataframe.drop_duplicates(subset=["UserID", "ItemID"], keep = "first", inplace = True)

        self._print("Loading Item Features Category")

        categories_dict = _read_categories_dict(zipFile_path + "catalog_ciao.txt")

        ICM_dataframe = dataframe[["ItemID", "CategoryID"]].copy()
        ICM_dataframe.drop_duplicates(keep = "first", inplace = True)
        ICM_dataframe.replace({"CategoryID": categories_dict}, inplace = True)
        ICM_dataframe["Data"] = 1
        ICM_dataframe.columns = ["ItemID", "FeatureID", "Data"]



        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_all_dataframe, "URM_all")
        dataset_manager.add_ICM(ICM_dataframe, "ICM_categories")

        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)

        self._print("Cleaning Temporary Files")

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        self._print("Loading Complete")

        return loaded_dataset


