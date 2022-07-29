#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Anonymous
"""

import zipfile, os, shutil
from Data_manager.DataReader import DataReader
import pandas as pd
from Data_manager.DatasetMapperManager import DatasetMapperManager



class TVAudienceReader(DataReader):

    DATASET_URL = "https://polimi365-my.sharepoint.com/:u:/g/personal/10322330_polimi_it/EZ1JPTmU6kRGnRezu3Ex-zQBxZK3-Y_aeP0Tb_3NbsQzHA?e=8YraeO"

    DATASET_SUBFOLDER = "TVAudience/"
    AVAILABLE_URM = ["URM_all", "URM_duration"]
    AVAILABLE_ICM = ["ICM_all", "ICM_genre", "ICM_channel", "ICM_subgenre", "ICM_event"]


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        self._print("Loading original data")

        compressed_zip_file_folder = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER
        decompressed_zip_file_folder = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        zipFile_name = "tv-audience-dataset.zip"

        try:

            dataFile = zipfile.ZipFile(compressed_zip_file_folder + zipFile_name)

            interactions_path = dataFile.extract("tv-audience-dataset/tv-audience-dataset.csv", path=decompressed_zip_file_folder + "decompressed/")

        except (FileNotFoundError, zipfile.BadZipFile):

            self._print("Unable to find or extract data zip file.")
            self._print("Automatic download not available, please ensure the ZIP data file is in folder {}.".format(compressed_zip_file_folder))
            self._print("Data zip file not found or damaged. You may download the data from: {}".format(self.DATASET_URL))

            # If directory does not exist, create
            if not os.path.exists(compressed_zip_file_folder):
                os.makedirs(compressed_zip_file_folder)

            raise FileNotFoundError("Automatic download not available.")


        self._print("Loading Interactions")

        """
        Columns are:
        channel ID: channel id from 1 to 217.
        slot: hour inside the week relative to the start of the view, from 1 to 24*7 = 168.
        week: week from 1 to 19. Weeks 14 and 19 should not be used because they contain errors.
        genre ID: it is the id of the genre, form 1 to 8. Genre/subgenre mapping is attached below.
        subGenre ID: it is the id of the subgenre, from 1 to 114. Genre/subgenre mapping is attached below.
        user ID: it is the id of the user.
        program ID: it is the id of the program. The same program can occur multiple times (e.g. a tv show).
        event ID: it is the id of the particular instance of a program. It is unique, but it can span multiple slots.
        duration: duration of the view.
        """

        columns = ["ChannelID", "Slot", "Week", "GenreID", "SubGenreID", "UserID", "ProgramID", "EventID", "Duration"]

        original_data_dataframe = pd.read_csv(filepath_or_buffer=interactions_path, sep=",", header=None,
                                        dtype= {index:str for index in range(len(columns)) if index != 8})

        original_data_dataframe.columns = columns

        URM_duration_dataframe = original_data_dataframe[["UserID", "ProgramID", "Duration"]]
        URM_duration_dataframe.columns = ["UserID", "ItemID", "Data"]

        URM_duration_dataframe = URM_duration_dataframe.groupby(['UserID', 'ItemID'], as_index=False )['Data'].sum()

        URM_all_dataframe = URM_duration_dataframe.copy()
        URM_all_dataframe["Data"] = 1

        def _create_single_feature_ICM(dataframe, column_name):
            ICM_dataframe = dataframe[["ProgramID", column_name]].copy()
            ICM_dataframe.columns = ["ItemID", "FeatureID"]
            ICM_dataframe["Data"] = 1
            ICM_dataframe = ICM_dataframe.groupby(['ItemID', 'FeatureID'], as_index=False )['Data'].max()
            ICM_dataframe["FeatureID"] = [column_name + "_" + str(x) for x in ICM_dataframe["FeatureID"]]

            return ICM_dataframe


        self._print("Loading Item Features")

        ICM_genre_dataframe = _create_single_feature_ICM(original_data_dataframe, "GenreID")
        ICM_channel_dataframe = _create_single_feature_ICM(original_data_dataframe, "ChannelID")
        ICM_subgenre_dataframe = _create_single_feature_ICM(original_data_dataframe, "SubGenreID")
        ICM_event_dataframe = _create_single_feature_ICM(original_data_dataframe, "EventID")

        ICM_all_dataframe = pd.concat([ICM_genre_dataframe, ICM_channel_dataframe,
                                       ICM_subgenre_dataframe, ICM_event_dataframe])


        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_all_dataframe, "URM_all")
        dataset_manager.add_URM(URM_duration_dataframe, "URM_duration")
        dataset_manager.add_ICM(ICM_genre_dataframe, "ICM_genre")
        dataset_manager.add_ICM(ICM_channel_dataframe, "ICM_channel")
        dataset_manager.add_ICM(ICM_subgenre_dataframe, "ICM_subgenre")
        dataset_manager.add_ICM(ICM_event_dataframe, "ICM_event")
        dataset_manager.add_ICM(ICM_all_dataframe, "ICM_all")

        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)


        self._print("Cleaning Temporary Files")

        shutil.rmtree(decompressed_zip_file_folder + "decompressed/", ignore_errors=True)

        self._print("Loading Complete")

        return loaded_dataset


