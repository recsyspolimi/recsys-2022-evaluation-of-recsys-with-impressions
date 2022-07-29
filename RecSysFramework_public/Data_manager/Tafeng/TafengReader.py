#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Anonymous
"""



import zipfile, shutil, os
import pandas as pd
from Data_manager.Dataset import Dataset
from Data_manager.DataReader import DataReader
from Data_manager.DatasetMapperManager import DatasetMapperManager
from Data_manager.DataReader_utils import download_from_URL


def _create_as_is_mapper(size):

    mapper = {}

    for index in range(size):
        mapper[index] = index

    return mapper

class TafengReader(DataReader):

    DATASET_URL = "https://sites.google.com/site/dataminingcourse2009/spring2016/annoucement2016/assignment3/D11-02.ZIP"
    DATASET_SUBFOLDER = "Tafeng/"
    AVAILABLE_URM = ["URM_all"]
    AVAILABLE_ICM = ["ICM_all", "ICM_classification", "ICM_cost"]
    AVAILABLE_UCM = ["UCM_all"]
    DATASET_SPECIFIC_MAPPER = []

    IS_IMPLICIT = True


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        self._print("Loading original data")

        zipFile_path =  self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "filmtrust.zip")

        except FileNotFoundError:

            self._print("Unable to find decompressed data file. Decompressing...")


            try:

                dataFile = zipfile.ZipFile(zipFile_path + "D11-02.zip")

            except (FileNotFoundError, zipfile.BadZipFile):
                self._print("Unable to find or extract data zip file. Downloading...")
                download_from_URL(self.DATASET_URL, zipFile_path, "D11-02.zip")

                dataFile = zipfile.ZipFile(zipFile_path + "D11-02.zip")


        self._print("Loading Interactions")
        all_dataframe = None

        for file_name in ["D01", "D02", "D11", "D12"]:
            URM_path = dataFile.extract(file_name, path=zipFile_path + "decompressed/")
            partial_dataframe = pd.read_csv(filepath_or_buffer=URM_path, sep=";", header=0, dtype={1:str, 2:str, 3:str, 4:str, 5:str})
            os.remove(URM_path)

            all_dataframe = partial_dataframe if all_dataframe is None else pd.concat([all_dataframe, partial_dataframe], axis=0)

        # URM_all_dataframe.columns = ["Transaction date", "customerID", "Age group", "Residence Area", "Product subclass", "Amount", "Asset", "Sales Price"]
        # 1: Transaction date and time (time invalid and useless)
        # 2: Customer ID
        # 3: Age: 10 possible values,
        # A <25,B 25-29,C 30-34,D 35-39,E 40-44,F 45-49,G 50-54,H 55-59,I 60-64,J >65
        # 4: Residence Area: 8 possible values,
        # A-F: zipcode area: 105,106,110,114,115,221,G: others, H: Unknown
        # Distance to store, from the closest: 115,221,114,105,106,110
        # 5: Product subclass
        # 6: Product ID
        # 7: Amount
        # 8: Asset
        # 9: Sales price

        all_dataframe.columns = ["Date", "Membership Card Number", "Age", "Region", "Product Classification", "Product Code", "Quantity", "Cost", "Sales"]

        URM_all_dataframe = all_dataframe[["Membership Card Number", "Product Code"]].copy()
        URM_all_dataframe.columns = ["UserID", "ItemID"]
        URM_all_dataframe.drop_duplicates(inplace=True)
        URM_all_dataframe["Data"] = 1

        self._print("Loading User Features")
        UCM_dataframe = all_dataframe[["Membership Card Number", "Age", "Region"]].copy()
        UCM_dataframe.columns = ["UserID", "Age", "Region"]
        UCM_dataframe = pd.melt(UCM_dataframe, id_vars="UserID", value_vars=["Age", "Region"], value_name='Data')
        UCM_dataframe['FeatureID'] = UCM_dataframe[['variable', 'Data']].apply('_'.join, axis=1)

        UCM_dataframe = UCM_dataframe[["UserID", "FeatureID", "Data"]]
        UCM_dataframe.drop_duplicates(inplace=True)
        UCM_dataframe["Data"] = 1


        self._print("Loading Item Features")
        ICM_classification_dataframe = all_dataframe[["Product Code", "Product Classification"]].copy()
        ICM_classification_dataframe.columns = ["ItemID", "FeatureID"]
        ICM_classification_dataframe.drop_duplicates(inplace=True)
        ICM_classification_dataframe["Data"] = 1

        ICM_cost_dataframe = all_dataframe[["Product Code", "Quantity", "Cost", "Sales"]].copy()
        ICM_cost_dataframe["Avg Cost"] = (ICM_cost_dataframe["Cost"]/ICM_cost_dataframe["Quantity"]).astype(int)
        # ICM_sales_dataframe = pd.melt(ICM_sales_dataframe, id_vars="Product Code", value_vars=["Quantity", "Cost", "Sales"], value_name='Data')
        ICM_cost_dataframe = ICM_cost_dataframe.groupby("Product Code")["Avg Cost"].mean()
        ICM_cost_dataframe = ICM_cost_dataframe.reset_index()
        ICM_cost_dataframe.columns = ["ItemID", "Data"]
        ICM_cost_dataframe["FeatureID"] = "Avg Cost"


        ICM_all_dataframe = pd.concat([ICM_classification_dataframe, ICM_cost_dataframe], axis=0, ignore_index=True)

        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_all_dataframe, "URM_all")
        dataset_manager.add_ICM(ICM_classification_dataframe, "ICM_classification")
        dataset_manager.add_ICM(ICM_cost_dataframe, "ICM_cost")
        dataset_manager.add_ICM(ICM_all_dataframe, "ICM_all")
        dataset_manager.add_UCM(UCM_dataframe, "UCM_all")

        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)

        self._print("Loading Complete")

        return loaded_dataset

