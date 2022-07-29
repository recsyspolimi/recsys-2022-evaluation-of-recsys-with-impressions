#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Anonymous
"""

from Data_manager.DataReader import DataReader
import zipfile, os, shutil
import pandas as pd
from Data_manager.DatasetMapperManager import DatasetMapperManager

class XingChallenge2016Reader(DataReader):

    DATASET_URL = "https://polimi365-my.sharepoint.com/:u:/g/personal/10322330_polimi_it/EcbIq2Iz731KnyWE9-CT-AYBPeIWINqWGMFC4t2TGpX9Tg?e=4nMDEK"

    DATASET_SUBFOLDER = "XingChallenge2016/"
    AVAILABLE_URM = ["URM_all", "URM_interaction_type", "URM_negative"]
    AVAILABLE_ICM = ["ICM_all"]

    IS_IMPLICIT = False

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        compressed_zip_file_folder = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER
        decompressed_zip_file_folder = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        zipFile_name = "xing_challenge_data_2016.zip"

        try:

            dataFile = zipfile.ZipFile(compressed_zip_file_folder + zipFile_name)

            # impressions_path = dataFile.extract("data/impressions.csv", path=decompressed_zip_file_folder + "decompressed/")
            interactions_path = dataFile.extract("data/interactions.csv", path=decompressed_zip_file_folder + "decompressed/")

            ICM_path = dataFile.extract("data/items.csv", path=decompressed_zip_file_folder + "decompressed/")
            # UCM_path = dataFile.extract("data/users.csv", path=decompressed_zip_file_folder + "decompressed/")

        except (FileNotFoundError, zipfile.BadZipFile):

            self._print("Unable to find or extract data zip file.")
            self._print("Automatic download not available, please ensure the ZIP data file is in folder {}.".format(compressed_zip_file_folder))
            self._print("Data zip file not found or damaged. You may download the data from: {}".format(self.DATASET_URL))

            # If directory does not exist, create
            if not os.path.exists(compressed_zip_file_folder):
                os.makedirs(compressed_zip_file_folder)

            raise FileNotFoundError("Automatic download not available.")




        self._print("Loading Interactions")
        URM_positive_dataframe, URM_interaction_type, URM_negative_dataframe = self._load_interactions(interactions_path)

        # print("XingChallenge2016Reader: Loading Impressions")
        # self.URM_impressions = self._load_impressions(impressions_path, if_new_user = "add", if_new_item = "add")

        self._print("Loading Item Features")
        ICM_metadata_dataframe = self._load_ICM(ICM_path)

        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_positive_dataframe, "URM_all")
        dataset_manager.add_URM(URM_interaction_type, "URM_interaction_type")
        dataset_manager.add_URM(URM_negative_dataframe, "URM_negative")
        dataset_manager.add_ICM(ICM_metadata_dataframe, "ICM_all")

        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)

        self._print("Cleaning Temporary Files")

        shutil.rmtree(decompressed_zip_file_folder + "decompressed/", ignore_errors=True)

        self._print("Loading Complete")

        return loaded_dataset




    def _load_interactions(self, impressions_path):
        """
        Interactions that the user performed on the job posting items. Fields:

        user_id             ID of the user who performed the interaction (points to users.id)
        item_id             ID of the item on which the interaction was performed (points to items.id)
        interaction_type    the type of interaction that was performed on the item:
            1 = the user clicked on the item
            2 = the user bookmarked the item on XING
            3 = the user clicked on the reply button or application form button that is shown on some job postings
            4 = the user deleted a recommendation from his/her list of recommendation (clicking on "x") which has the effect that the recommendation will no longer been shown to the user and that a new recommendation item will be loaded and displayed to the user
        created_at          a unix time stamp timestamp representing the time when the interaction got created
        """

        columns = ["user_id", "item_id", "interaction_type", "created_at"]

        URM_interaction_type = pd.read_csv(filepath_or_buffer=impressions_path, sep="\t", header=0,
                                    usecols = columns, dtype= {0:str, 1:str, 2:int, 3:float})

        URM_interaction_type.columns = ["UserID", "ItemID", "Data", "Timestamp"]
        URM_interaction_type = URM_interaction_type[["UserID", "ItemID", "Data"]]

        URM_negative_dataframe = URM_interaction_type.copy()
        URM_negative_dataframe = URM_negative_dataframe[URM_negative_dataframe["Data"] == 4]

        URM_positive_dataframe = URM_interaction_type[URM_interaction_type["Data"] != 4].copy()
        URM_positive_dataframe["Data"] = 1

        return URM_positive_dataframe, URM_interaction_type, URM_negative_dataframe




    def _load_ICM(self, ICM_path):

        """
        id              anonymized ID of the item (referenced as item_id in the other datasets above)
        title           concepts that have been extracted from the job title of the job posting (numeric IDs)
        career_level    career level ID (e.g. beginner, experienced, manager):
            0 = unknown
            1 = Student/Intern
            2 = Entry Level (Beginner)
            3 = Professional/Experienced
            4 = Manager (Manager/Supervisor)
            5 = Executive (VP, SVP, etc.)
            6 = Senior Executive (CEO, CFO, President)
        discipline_id   anonymized IDs represent disciplines such as "Consulting", "HR", etc.
        industry_id     anonymized IDs represent industries such as "Internet", "Automotive", "Finance", etc.
        country         code of the country in which the job is offered
        region          is specified for some users who have as country de. Meaning of the regions: see below.
        latitude        latitude information (rounded to ca. 10km)
        longitude       longitude information (rounded to ca. 10km)
        employment      the type of employment:
            0 = unknown
            1 = full-time
            2 = part-time
            3 = freelancer
            4 = intern
            5 = voluntary
        tags            concepts that have been extracted from the tags, skills or company name
        created_at      a Unix time stamp timestamp representing the time when the interaction got created
        active_during_test is 1 if the item is still active (= recommendable) during the test period and 0 if the item is not active anymore in the test period (= not recommendable)
        """


        columns = ["id","title","career_level","discipline_id","industry_id",
                   "country","region","latitude","longitude",
                   "employment","tags","created_at","active_during_test"]

        ICM_metadata_dataframe = pd.read_csv(filepath_or_buffer=ICM_path, sep="\t", header=0,
                                    usecols = columns, dtype= {index:str for index in range(len(columns))})


        # Split data into lists and replace Nans with empty lists to match the data type of other values in same column
        for column_type in ["title", "discipline_id", "industry_id", "tags"]:
            ICM_metadata_dataframe[column_type] = ICM_metadata_dataframe[column_type].str.split(",")
            isnull = ICM_metadata_dataframe[column_type].isnull()
            ICM_metadata_dataframe.loc[isnull, column_type] = [[] * isnull.sum()]

        for column_type in ["career_level", "country", "region", "employment"]:
            ICM_metadata_dataframe[column_type] = column_type + "_" + ICM_metadata_dataframe[column_type]


        item_id_list = []
        token_lists = []

        for index, data_row in ICM_metadata_dataframe.iterrows():

            if (index+1) % 100000 == 0 or (index +1) == len(ICM_metadata_dataframe):
                print("Processed [{}/{}] rows".format(index+1, len(ICM_metadata_dataframe)))

            job_title_id_list = ["title_" + str(ID) for ID in data_row["title"]]

            career_level = data_row["career_level"]

            discipline_id = ["discipline_id_" + ID for ID in data_row["discipline_id"]]
            industry_id = ["industry_id_" + ID for ID in data_row["industry_id"]]

            country = data_row["country"]
            region = data_row["region"]
            employment = data_row["employment"]

            tags_list = ["tags_" + ID for ID in data_row["tags"]]

            # created_at = "created_at_" + str(data_row["created_at"])
            # active_during_test = "active_during_test_" + str(data_row["active_during_test"])

            this_token_list = [*job_title_id_list, career_level, *discipline_id, *industry_id,
                               country, region, employment, *tags_list]

            item_id_list.append(data_row["id"])
            token_lists.append(this_token_list)



        ICM_metadata_dataframe = pd.DataFrame(token_lists, index=item_id_list).stack()
        ICM_metadata_dataframe = ICM_metadata_dataframe.reset_index()[["level_0", 0]]
        ICM_metadata_dataframe.columns = ['ItemID', 'FeatureID']
        ICM_metadata_dataframe["Data"] = 1

        return ICM_metadata_dataframe


