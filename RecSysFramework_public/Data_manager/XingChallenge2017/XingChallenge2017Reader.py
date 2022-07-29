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


class XingChallenge2017Reader(DataReader):

    DATASET_URL = "https://polimi365-my.sharepoint.com/:u:/g/personal/10322330_polimi_it/EYtfBPdYbbxAjPc8hvi5vCgBVXEEM5s5R6zwHC0JAPGG4w?e=OORsFX"
    DATASET_SUBFOLDER = "XingChallenge2017/"
    AVAILABLE_URM = ["URM_all", "URM_interaction_type", "URM_negative", "URM_impressions"]
    AVAILABLE_ICM = ["ICM_all"]

    IS_IMPLICIT = False

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        self._print("Loading original data")


        compressed_zip_file_folder = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER
        decompressed_zip_file_folder = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        zipFile_name = "xing_challenge_data_2017.zip"


        try:

            dataFile = zipfile.ZipFile(compressed_zip_file_folder + zipFile_name)

            interactions_path = dataFile.extract("data/interactions_14.csv", path=decompressed_zip_file_folder + "decompressed/")

            ICM_path = dataFile.extract("data/items.csv", path=decompressed_zip_file_folder + "decompressed/")
            #UCM_path = dataFile.extract("data/users.csv", path=decompressed_zip_file_folder + "decompressed/")

        except (FileNotFoundError, zipfile.BadZipFile):

            self._print("Unable to find or extract data zip file.")
            self._print("Automatic download not available, please ensure the ZIP data file is in folder {}.".format(compressed_zip_file_folder))
            self._print("Data zip file not found or damaged. You may download the data from: {}".format(self.DATASET_URL))

            # If directory does not exist, create
            if not os.path.exists(compressed_zip_file_folder):
                os.makedirs(compressed_zip_file_folder)

            raise FileNotFoundError("Automatic download not available.")


        self._print("Loading Interactions")
        URM_positive_dataframe, URM_interaction_type, URM_negative_dataframe, URM_impressions_dataframe = self._load_interactions(interactions_path)


        # print("XingChallenge2017Reader: Loading Impressions")
        # self.URM_impressions = self._load_impressions(impressions_path, if_new_user = "add", if_new_item = "add")

        self._print("Loading Item Features")
        ICM_metadata_dataframe = self._load_ICM(ICM_path)


        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_positive_dataframe, "URM_all")
        dataset_manager.add_URM(URM_interaction_type, "URM_interaction_type")
        dataset_manager.add_URM(URM_negative_dataframe, "URM_negative")
        dataset_manager.add_URM(URM_impressions_dataframe, "URM_impressions")
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

        user_id ID          of the user who performed the interaction (points to users.id)
        item_id ID          of the item on which the interaction was performed (points to items.id)
        created_at          a unix time stamp timestamp representing the time when the interaction got created
        interaction_type    the type of interaction that was performed on the item:
            0 = XING showed this item to a user (= impression)
            1 = the user clicked on the item
            2 = the user bookmarked the item on XING
            3 = the user clicked on the reply button or application form button that is shown on some job postings
            4 = the user deleted a recommendation from his/her list of recommendation (clicking on "x") which has the effect that the recommendation will no longer been shown to the user and that a new recommendation item will be loaded and displayed to the user
            5 = (not used) a recruiter from the items company showed interest into the user. (e.g. clicked on the profile)

        """

        columns = ["user_id", "item_id", "interaction_type", "created_at"]

        URM_interaction_type = pd.read_csv(filepath_or_buffer=impressions_path, sep="\t", header=0,
                                    usecols = columns, dtype= {0:str, 1:str, 2:int, 3:float})

        URM_interaction_type.columns = ["UserID", "ItemID", "Data", "Timestamp"]
        URM_interaction_type = URM_interaction_type[URM_interaction_type["Data"]!=5]
        URM_interaction_type = URM_interaction_type[["UserID", "ItemID", "Data"]]

        URM_negative_dataframe = URM_interaction_type.copy()
        URM_negative_dataframe = URM_negative_dataframe[URM_negative_dataframe["Data"] == 4]

        URM_impressions_dataframe = URM_interaction_type.copy()
        URM_impressions_dataframe = URM_impressions_dataframe[URM_impressions_dataframe["Data"] == 0]

        URM_positive_dataframe = URM_interaction_type.loc[(URM_interaction_type["Data"] >= 1) & (URM_interaction_type["Data"] <= 3)].copy()
        URM_positive_dataframe["Data"] = 1

        return URM_positive_dataframe, URM_interaction_type, URM_negative_dataframe, URM_impressions_dataframe



    def _load_ICM(self, ICM_path):
        """
        ORDERING IN CSV FILE
        # item_id
        title
        career_level
        discipline_id
        industry_id
        country
        is_payed
        region
        latitude
        longitude
        employment
        tags
        created_at


        id anonymized ID        of the item (referenced as item_id in the other datasets above)
        industry_id             anonymized IDs represent industries such as "Internet", "Automotive", "Finance", etc.
        discipline_id           anonymized IDs represent disciplines such as "Consulting", "HR", etc.
        is_paid (or is_payed)   indicates that the posting is a paid for by a compnay
        career_level            career level ID (e.g. beginner, experienced, manager)
            0 = unknown
            1 = Student/Intern
            2 = Entry Level (Beginner)
            3 = Professional/Experienced
            4 = Manager (Manager/Supervisor)
            5 = Executive (VP, SVP, etc.)
            6 = Senior Executive (CEO, CFO, President)
        country                 code of the country in which the job is offered
        latitude                latitude information (rounded to ca. 10km)
        longitude               longitude information (rounded to ca. 10km)
        region                  is specified for some users who have as country `de`. Meaning of the regions: see below.
        employment              the type of emploment
            0 = unknown
            1 = full-time
            2 = part-time
            3 = freelancer
            4 = intern
            5 = voluntary
        created_at              a unix time stamp timestamp representing the time when the interaction got created
        title                   concepts that have been extracted from the job title of the job posting (numeric IDs)
        tags                    concepts that have been extracted from the tags, skills or company name
        """

        columns = ["item_id", "title","career_level","discipline_id","industry_id","country", "is_payed",
                   "region","latitude","longitude",
                   "employment","tags","created_at"]

        ICM_metadata_dataframe = pd.read_csv(filepath_or_buffer=ICM_path, sep="\t", header=0,
                                    usecols = columns, dtype= {index:str for index in range(len(columns))})


        # Split data into lists and replace Nans with empty lists to match the data type of other values in same column
        for column_type in ["title", "discipline_id", "industry_id", "tags"]:
            ICM_metadata_dataframe[column_type] = ICM_metadata_dataframe[column_type].str.split(",")
            isnull = ICM_metadata_dataframe[column_type].isnull()
            ICM_metadata_dataframe.loc[isnull, column_type] = [[] * isnull.sum()]


        for column_type in ["career_level", "country", "is_payed", "region", "employment"]:
            ICM_metadata_dataframe[column_type] = column_type + "_" + ICM_metadata_dataframe[column_type]

        item_id_list = []
        token_lists = []

        for index, data_row in ICM_metadata_dataframe.iterrows():

            if (index+1) % 100000 == 0 or (index +1) == len(ICM_metadata_dataframe):
                print("Processed [{}/{}] rows".format(index+1, len(ICM_metadata_dataframe)))

            job_title_id_list = ["title_" + ID for ID in data_row["title"]]

            career_level = data_row["career_level"]

            discipline_id = ["discipline_id_" + ID for ID in data_row["discipline_id"]]
            industry_id = ["industry_id_" + ID for ID in data_row["industry_id"]]

            country = data_row["country"]
            is_payed = data_row["is_payed"]
            region = data_row["region"]
            # latitude = "latitude_" + str(data_row["latitude"])
            # longitude = "longitude_" + str(data_row["longitude"])
            employment = data_row["employment"]

            tags_list = ["tags_" + str(ID) for ID in data_row["tags"]]

            # created_at = "created_at_" + str(data_row["created_at"])

            this_token_list = [*job_title_id_list, career_level, *discipline_id, *industry_id, country, is_payed, region, employment, *tags_list]

            item_id_list.append(data_row["item_id"])
            token_lists.append(this_token_list)



        ICM_metadata_dataframe = pd.DataFrame(token_lists, index=item_id_list).stack()
        ICM_metadata_dataframe = ICM_metadata_dataframe.reset_index()[["level_0", 0]]
        ICM_metadata_dataframe.columns = ['ItemID', 'FeatureID']
        ICM_metadata_dataframe["Data"] = 1

        return ICM_metadata_dataframe
