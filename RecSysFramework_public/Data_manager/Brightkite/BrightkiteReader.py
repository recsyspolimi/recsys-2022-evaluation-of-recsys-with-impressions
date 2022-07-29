#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Anonymous
"""


import gzip, os
import pandas as pd
from Data_manager.DatasetMapperManager import DatasetMapperManager
from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import download_from_URL


class BrightkiteReader(DataReader):

    DATASET_URL = "https://snap.stanford.edu/data/loc-brightkite_edges.txt.gz"
    DATASET_SUBFOLDER = "Brightkite/"
    AVAILABLE_ICM = []



    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        self._print("Loading original data")

        folder_path = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER


        try:

            compressed_file = gzip.open(folder_path + "loc-brightkite_edges.txt.gz", 'rb')

        except (FileNotFoundError):

            self._print("Unable to find or extract data zip file. Downloading...")

            download_from_URL(self.DATASET_URL, folder_path, "loc-brightkite_edges.txt.gz")

            compressed_file = gzip.open(folder_path + "loc-brightkite_edges.txt.gz", 'rb')

        URM_path = folder_path + "loc-brightkite_edges.txt"

        decompressed_file = open(URM_path, "w")

        self._save_GZ_in_text_file(compressed_file, decompressed_file)

        decompressed_file.close()

        self._print("Loading Interactions")
        URM_all_dataframe = pd.read_csv(filepath_or_buffer=URM_path, sep="\t", header=None, dtype={0:str, 1:str})
        URM_all_dataframe.columns = ["UserID", "ItemID"]
        URM_all_dataframe["Data"] = 1

        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_all_dataframe, "URM_all")

        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)

        self._print("Cleaning Temporary Files")

        os.remove(URM_path)

        self._print("Loading Complete")

        return loaded_dataset



    def _save_GZ_in_text_file(self, compressed_file, decompressed_file):

        print("BrightkiteReader: decompressing file...")

        for line in compressed_file:
            decompressed_file.write(line.decode("utf-8"))

        decompressed_file.flush()

        print("BrightkiteReader: decompressing file... done!")

