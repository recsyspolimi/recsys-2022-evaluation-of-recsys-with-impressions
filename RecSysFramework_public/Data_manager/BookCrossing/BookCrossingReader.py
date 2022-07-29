#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Anonymous
"""


import zipfile, shutil, csv
import pandas as pd
from Data_manager.DatasetMapperManager import DatasetMapperManager
from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import download_from_URL


class BookCrossingReader(DataReader):
    """
    Collected from: http://www2.informatik.uni-freiburg.de/~cziegler/BX/

    Ratings are in the range 1-10

    """

    DATASET_URL = "http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip"
    DATASET_SUBFOLDER = "BookCrossing/"
    AVAILABLE_URM = ["URM_all", "URM_implicit", "URM_rating"]
    AVAILABLE_ICM = ["ICM_book_crossing"]

    IS_IMPLICIT = False


    def __init__(self, **kwargs):
        super(BookCrossingReader, self).__init__(**kwargs)
        self._print("ICM contains the author, publisher, year and tokens from the title")


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        self._print("Loading original data")

        folder_path = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER


        try:

            dataFile = zipfile.ZipFile(folder_path + "BX-CSV-Dump.zip")

        except (FileNotFoundError, zipfile.BadZipFile):
            self._print("Unable to find or extract data zip file. Downloading...")
            try:
                download_from_URL(self.DATASET_URL, folder_path, "BX-CSV-Dump.zip")
            except:
                folder_path = self.DATASET_OFFLINE_ROOT_FOLDER + "BookCrossing/"
            dataFile = zipfile.ZipFile(folder_path + "BX-CSV-Dump.zip")


        URM_path = dataFile.extract("BX-Book-Ratings.csv", path=folder_path + "decompressed")
        ICM_path = dataFile.extract("BX-Books.csv", path=folder_path + "decompressed")


        self._print("Loading Item Features")
        ICM_book_crossing = pd.read_csv(filepath_or_buffer=ICM_path, sep='";"', header=0, dtype={index:str for index in range(8)}, engine='python')
        ICM_book_crossing = ICM_book_crossing.rename(columns={'"ISBN': 'ISBN'})
        ICM_book_crossing = ICM_book_crossing.rename(columns={'Image-URL-L"': 'Image-URL-L'})

        ICM_book_crossing['ISBN'] = ICM_book_crossing['ISBN'].str.replace('"','')
        ICM_book_crossing['Image-URL-L'] = ICM_book_crossing['Image-URL-L'].str.replace('"','')

        from Data_manager.TagPreprocessing import tagFilterAndStemming

        title_tag_list = [tagFilterAndStemming(title) for title in ICM_book_crossing["Book-Title"]]
        other_tag_list = [list(x) for x in zip(ICM_book_crossing['Book-Author'], ICM_book_crossing['Year-Of-Publication'], ICM_book_crossing['Publisher'])]

        ICM_book_crossing["TagList"] = [list([*x, *y]) for x,y in zip(title_tag_list, other_tag_list)]

        ICM_book_crossing = ICM_book_crossing[["ISBN", "TagList"]]
        ICM_book_crossing.columns = ["ItemID", "TagList"]

        # Split TagList in order to obtain a dataframe with a tag per row
        ICM_book_crossing = pd.DataFrame(ICM_book_crossing["TagList"].tolist(), index=ICM_book_crossing["ItemID"]).stack()
        ICM_book_crossing = ICM_book_crossing.reset_index()[[0, 'ItemID']]
        ICM_book_crossing.columns = ['FeatureID', 'ItemID']
        ICM_book_crossing["Data"] = 1


        #############################
        ##########
        ##########      Load metadata using AmazonReviewData
        ##########      for books ASIN corresponds to ISBN
        #
        # print("BookCrossingReader: loading ICM from AmazonReviewData")
        #
        # from Data_manager.AmazonReviewData._AmazonReviewDataReader import _AmazonReviewDataReader
        #
        # # Pass "self" object as it contains the item_id mapper already initialized with the ISBN
        # ICM_amazon, tokenToFeatureMapper_ICM_amazon = _AmazonReviewDataReader._loadMetadata(self, if_new_item ="add")
        #
        # ICM_amazon, _, tokenToFeatureMapper_ICM_amazon = remove_features(ICM_amazon, minOccurrence = 5, maxPercOccurrence = 0.30,
        #                                                                           reconcile_mapper=tokenToFeatureMapper_ICM_amazon)
        #
        # self.loaded_ICM_dict["ICM_amazon"] = ICM_amazon
        # self.loaded_ICM_mapper_dict["ICM_amazon"] = tokenToFeatureMapper_ICM_amazon

        self._print("Loading Interactions")
        URM_all_dataframe = pd.read_csv(filepath_or_buffer=URM_path, sep=';', header=0,
                                        encoding='latin1', quotechar = '"', quoting=csv.QUOTE_ALL,
                                        dtype={0:str, 1:str, 3:int})

        URM_all_dataframe.columns = ["UserID", "ItemID", "Data"]

        # Split interactions in explicit and implicit
        URM_rating_dataframe = URM_all_dataframe[URM_all_dataframe["Data"] != 0].copy()
        URM_implicit_dataframe = URM_all_dataframe[URM_all_dataframe["Data"] == 0].copy()
        URM_implicit_dataframe["Data"] = 1.0

        URM_all_dataframe["Data"] = 1.0

        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_all_dataframe, "URM_all")
        dataset_manager.add_URM(URM_rating_dataframe, "URM_rating")
        dataset_manager.add_URM(URM_implicit_dataframe, "URM_implicit")
        dataset_manager.add_ICM(ICM_book_crossing, "ICM_book_crossing")

        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)

        self._print("Cleaning Temporary Files")

        shutil.rmtree(folder_path + "decompressed", ignore_errors=True)

        self._print("Loading Complete")

        return loaded_dataset
