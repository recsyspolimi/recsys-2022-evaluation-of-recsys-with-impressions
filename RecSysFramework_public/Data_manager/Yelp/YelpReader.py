#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10/01/18

@author: Anonymous
"""


import ast, tarfile, os, shutil
import pandas as pd
from Data_manager.DatasetMapperManager import DatasetMapperManager
from Data_manager.DataReader import DataReader



def parse_json(file_path):
    g = open(file_path, 'r', encoding="latin1")

    for l in g:
        try:
            yield ast.literal_eval(l)
        except Exception as exception:
            print("Exception: {}. Skipping".format(str(exception)))




class YelpReader(DataReader):
    """
    Documentation here: https://www.yelp.com/dataset/documentation/main
    """
    #DATASET_URL = "https://www.yelp.com/dataset"
    DATASET_URL = "https://polimi365-my.sharepoint.com/:u:/g/personal/10322330_polimi_it/EXwcwdmew5dLgmbCHMXyX-4B0rGDMH1qIpZEo4OS0xCZ-w?e=knMda2"
    DATASET_SUBFOLDER = "Yelp/"
    AVAILABLE_ICM = []

    IS_IMPLICIT = False

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):

        # Load data from original

        self._print("Loading original data")

        compressed_file_folder = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER
        decompressed_file_folder = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER


        try:

            compressed_file = tarfile.open(compressed_file_folder + "yelp_dataset.tar", "r")
            compressed_file.extract("yelp_academic_dataset_review.json", path=decompressed_file_folder + "decompressed/")
            compressed_file.close()

        except (FileNotFoundError, tarfile.ReadError, tarfile.ExtractError):

            self._print("Unable to fild or decompress tar.gz file.")
            self._print("Automatic download not available, please ensure the ZIP data file is in folder {}.".format(compressed_file_folder))
            self._print("Data can be downloaded here: {}".format(self.DATASET_URL))

            # If directory does not exist, create
            if not os.path.exists(compressed_file_folder):
                os.makedirs(compressed_file_folder)

            raise FileNotFoundError("Automatic download not available.")




        URM_path = decompressed_file_folder + "decompressed/yelp_academic_dataset_review.json"

        self._print("Loading Interactions")
        URM_all_dataframe = self._loadURM(URM_path)

        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_all_dataframe, "URM_all")


        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)

        self._print("Cleaning Temporary Files")

        shutil.rmtree(decompressed_file_folder + "decompressed/", ignore_errors=True)

        self._print("Loading Complete")

        return loaded_dataset









    def _loadURM (self, filePath):

        parser_metadata = parse_json(filePath)

        business_id_list = []
        user_id_list = []
        rating_list = []

        for index, new_metadata in enumerate(parser_metadata):

            if index % 1000000 == 0 and index>0:
                print("Processed {}".format(index))

            business_id_list.append(new_metadata["business_id"])
            user_id_list.append(new_metadata["user_id"])
            rating_list.append(float(new_metadata["stars"]))

        URM_all_dataframe = pd.DataFrame({"UserID": user_id_list,
                                          "ItemID": business_id_list,
                                          "Data":  rating_list,
                                          })

        return  URM_all_dataframe


