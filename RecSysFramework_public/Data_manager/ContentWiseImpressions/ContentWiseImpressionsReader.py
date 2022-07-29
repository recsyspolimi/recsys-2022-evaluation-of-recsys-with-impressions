#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 09/07/21

@author: Anonymous
"""

from Data_manager.DataReader import DataReader
import zipfile, os, shutil, gzip
import pandas as pd
from ast import literal_eval
from Data_manager.DatasetMapperManager import DatasetMapperManager


def _decompress_internal_gzip(dataFile, decompressed_zip_file_folder, internal_CSV_path, gz_file_name):

    interactions_path = dataFile.extract(internal_CSV_path + gz_file_name, path=decompressed_zip_file_folder)
    compressed_file = gzip.open(interactions_path, "rb")

    csv_file_name = gz_file_name.rstrip(".gz")
    decompressed_file = open(decompressed_zip_file_folder + csv_file_name, "wb")
    decompressed_file.write(compressed_file.read())

    compressed_file.close()
    decompressed_file.close()

    return decompressed_zip_file_folder + csv_file_name


class ContentWiseImpressionsReader(DataReader):
    """Dataset published by ContentWise on CIKM2020.

    This dataset contains interactions and impressions from an Over-the-Top media service. The dataset contains
    10 million interactions of users with items related to the cinema and television. It also has impressions of
    items presented to the users, the length of recommendations, row_position. We collected two different types of
    impressions: impressions with and without direct links to interactions.

    The code contains methods to read the dataset from local disk, generate URM matrices, training/validation/testing splits,
     save them URMs to disk, and more.

    This dataset was published in the article "ContentWise Impressions: An industrial dataset with impressions included"
    by F.B. Pérez Maurera, Maurizio Ferrari Dacrema, Lorenzo Saule, Mario Scriminaci, and Paolo Cremonesi. If you use
    this code or the dataset, please reference our work.

    @Article{ContentWiseImpressions,
        author={Pérez Maurera, Fernando Benjamín
            and Ferrari Dacrema, Maurizio
            and Saule, Lorenzo
            and Scriminaci, Mario
            and Cremonesi, Paolo},
        title={ContentWise Impressions: An industrial dataset with impressions included},
        journal={Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM 2020)},
        year={2020},
        doi={},
        Eprint={arXiv},
        note={Source: \\url{https://github.com/ContentWise/contentwise-impressions}},
    }


    The following are the types presented in the dataset.
    Interaction columns
    --------
    utc_ts_milliseconds (index): int64
        UTC Unix timestamps of interactions.
    user_id: int32.
        Anonymized identifier of users.
    item_id: int32.
        Anonymized identifier of items.
    item_type`: int8
        Classification of the item. It has 4 possible values.
            0: Movies.
            1: Movies and clips in series.
            2: TV Movies or shows.
            3: Episodes of TV Series.
    series_id: int32.
        Anonymized identifier of series.
    episode_number: int32.
        Episode number of the item inside a series.
    series_length: int32.
        Number of episodes of the series.
    recommendation_id: int32.
        Identifier of recommendation presented to the user.
    interaction_type`: int8
        Classification of the interaction. It has 4 possible values.
            0: The user viewed an item.
            1: The user accessed an item.
            2: The user rated an item.
            3: The user purchased an item.
    explicit_rating: float32
        Rating that the user gave to an item. Ranges from 0 to 5 with steps of 0.5
    vision_factor: float32
        Reflects how much a user viewed an item based on the item's duration. Ranges from 0 to 1.


    Impressions with direct links columns
    --------
    recommendation_id : int32.
        (index) Anonymized identifier of recommendation presented to the user.
    row_position : int32
        Position on screen of recommendation.
    recommendation_list_length : int32
        Number of recommended items.
    recommended_series_list : List[int32]
        Ordered recommendation list of series ids. Series on the first positions are considered to be more
        meaningful to users.


    Impressions without direct links columns
    --------
    user_id : int32.
        (index) Anonymized user identifier that received the recommendations.
    row_position : int32
        Position on screen of recommendation.
    recommendation_list_length : int32
        Number of recommended items.
    recommended_series_list : List[int32]
        Ordered recommendation list of series ids. Series on the first positions are considered to be more more
        meaningful to users.
    """

    DATASET_URL = "https://polimi365-my.sharepoint.com/:u:/g/personal/10322330_polimi_it/EYb8Gl_onbxEhfCIJc_JHbMBSvJGw43vdP0diZW-_d7ttw?e=uBiMN4"
    DATASET_SUBFOLDER = "ContentWiseImpressions/"
    AVAILABLE_URM = ["URM_all"] #, "URM_impressions_non_direct"]
    AVAILABLE_ICM = ["ICM_series", "ICM_type", "ICM_all"]

    IS_IMPLICIT = False

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        compressed_zip_file_folder = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER
        decompressed_zip_file_folder = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        zipFile_name = "ContentWiseImpressionsData.zip"

        try:

            dataFile = zipfile.ZipFile(compressed_zip_file_folder + zipFile_name)

            internal_CSV_path = "ContentWiseImpressions/data/ContentWiseImpressions/CW10M-CSV/"

            # impressions_direct_path = _decompress_internal_gzip(dataFile, decompressed_zip_file_folder + "decompressed/", internal_CSV_path, "impressions-direct-link.csv.gz")
            # impressions_non_direct_path = _decompress_internal_gzip(dataFile, decompressed_zip_file_folder + "decompressed/", internal_CSV_path, "impressions-non-direct-link.csv.gz")
            interactions_path = _decompress_internal_gzip(dataFile, decompressed_zip_file_folder + "decompressed/", internal_CSV_path, "interactions.csv.gz")

        except (FileNotFoundError, zipfile.BadZipFile):

            self._print("Unable to find or extract data zip file.")
            self._print("Automatic download not available, please ensure the ZIP data file is in folder {}.".format(compressed_zip_file_folder))
            self._print("Data zip file not found or damaged. You may download the data from: {}".format(self.DATASET_URL))

            # If directory does not exist, create
            if not os.path.exists(compressed_zip_file_folder):
                os.makedirs(compressed_zip_file_folder)

            raise FileNotFoundError("Automatic download not available.")




        self._print("Loading Interactions")

        interactions_dataframe = pd.read_csv(interactions_path, sep=",", header=0, index_col=False,
                                                 dtype={"user_id":str, "item_id":str, "series_id":str, "item_type":str})

        # URM_all_dataframe = interactions_dataframe[["user_id", "item_id", "interaction_type"]].copy()
        URM_all_dataframe = interactions_dataframe[["user_id", "item_id"]].drop_duplicates()
        URM_all_dataframe.columns = ["UserID", "ItemID"]
        URM_all_dataframe["Data"] = 1


        # self._print("Loading Impressions")
        # impressions_direct_dataframe = pd.read_csv(impressions_direct_path, sep=",", header=0, index_col=False)
        # impressions_non_direct_dataframe = pd.read_csv(impressions_non_direct_path, sep=",", header=0, index_col=False)
        #
        # impressions_direct_dataframe_joined = impressions_direct_dataframe.set_index('recommendation_id').join(interactions_dataframe.set_index('recommendation_id'), how='left', on="recommendation_id")
        #
        # def _string_to_list(data_series):
        #     data_series = data_series.str.replace("\n", "", regex=False)
        #     data_series = data_series.str.replace("\[ +", "[", regex=True)
        #     data_series = data_series.str.replace(" +", ",", regex=True)
        #     return data_series.apply(literal_eval)
        #
        # impressions_direct_dataframe["recommended_series_list"] = _string_to_list(impressions_direct_dataframe["recommended_series_list"])
        # impressions_non_direct_dataframe["recommended_series_list"] = _string_to_list(impressions_non_direct_dataframe["recommended_series_list"])
        #
        # URM_impressions_non_direct_dataframe = impressions_non_direct_dataframe[["user_id", "recommended_series_list"]]
        # URM_impressions_non_direct_dataframe = URM_impressions_non_direct_dataframe.explode("recommended_series_list").astype(str)
        # URM_impressions_non_direct_dataframe.columns = ["UserID", "ItemID"]
        # URM_impressions_non_direct_dataframe["Data"] = 1


        self._print("Loading Item Features")
        ICM_series_dataframe = interactions_dataframe[["item_id", "series_id"]].drop_duplicates()

        ICM_type_dataframe = interactions_dataframe[["item_id", "item_type"]].drop_duplicates()
        item_type_dict = {"0": "Movies",
                          "1": "Movies and clips in series",
                          "2": "TV Movies or shows",
                          "3": "Episodes of TV Series"}
        ICM_type_dataframe.replace({"item_type": item_type_dict}, inplace = True)


        ICM_series_dataframe.columns = ["ItemID", "FeatureID"]
        ICM_series_dataframe["Data"] = 1

        ICM_type_dataframe.columns = ["ItemID", "FeatureID"]
        ICM_type_dataframe["Data"] = 1

        ICM_all_dataframe = pd.concat([ICM_series_dataframe, ICM_type_dataframe], axis=0)

        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_all_dataframe, "URM_all")
        # dataset_manager.add_URM(URM_impressions_non_direct_dataframe, "URM_impressions_non_direct")
        dataset_manager.add_ICM(ICM_series_dataframe, "ICM_series")
        dataset_manager.add_ICM(ICM_type_dataframe, "ICM_type")
        dataset_manager.add_ICM(ICM_all_dataframe, "ICM_all")

        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)

        self._print("Cleaning Temporary Files")

        shutil.rmtree(decompressed_zip_file_folder + "decompressed/", ignore_errors=True)

        self._print("Loading Complete")

        return loaded_dataset



