#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Anonymous
"""

import zipfile, re
import numpy as np
import ast, csv, os, shutil
from Data_manager.DataReader import DataReader
from Data_manager.Movielens._utils_movielens_parser import _loadURM

import pandas as pd
from Data_manager.DatasetMapperManager import DatasetMapperManager

class TheMoviesDatasetReader(DataReader):

    #DATASET_URL = "https://www.kaggle.com/rounakbanik/the-movies-dataset"
    DATASET_URL = "https://polimi365-my.sharepoint.com/:u:/g/personal/10322330_polimi_it/EQAIIMiVSTpIjZYmDqMyvukB8dur9LJ5cRT83CzXpLZ0TQ?e=lRNtWF"
    DATASET_SUBFOLDER = "TheMoviesDataset/"
    AVAILABLE_URM = ["URM_all", "URM_timestamp"]
    AVAILABLE_ICM = ["ICM_all", "ICM_credits", "ICM_metadata"]
    DATASET_SPECIFIC_MAPPER = ["item_original_ID_to_title", "item_index_to_title"]

    IS_IMPLICIT = False

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        self._print("Loading original data")

        compressed_zip_file_folder = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER
        decompressed_zip_file_folder = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        zipFile_name = "the-movies-dataset.zip"


        try:

            dataFile = zipfile.ZipFile(compressed_zip_file_folder + zipFile_name)

            credits_path = dataFile.extract("credits.csv", path=decompressed_zip_file_folder + "decompressed/")
            metadata_path = dataFile.extract("movies_metadata.csv", path=decompressed_zip_file_folder + "decompressed/")
            movielens_tmdb_id_map_path = dataFile.extract("links.csv", path=decompressed_zip_file_folder + "decompressed/")

            URM_path = dataFile.extract("ratings.csv", path=decompressed_zip_file_folder + "decompressed/")

        except (FileNotFoundError, zipfile.BadZipFile):

            self._print("Unable to find or extract data zip file.")
            self._print("Automatic download not available, please ensure the ZIP data file is in folder {}.".format(compressed_zip_file_folder))
            self._print("Data zip file not found or damaged. You may download the data from: {}".format(self.DATASET_URL))

            # If directory does not exist, create
            if not os.path.exists(compressed_zip_file_folder):
                os.makedirs(compressed_zip_file_folder)

            raise FileNotFoundError("Automatic download not available.")


        self._print("Loading Item Features Credits")
        ICM_credits_dataframe = self._loadICM_credits(credits_path, header=True)

        self._print("Loading Item Features Metadata")
        ICM_metadata_dataframe = self._loadICM_metadata(metadata_path)

        # IMPORTANT: ICM uses TMDB indices, URM uses movielens indices
        # Convert TMDB indices in movielens ones directly in the ICMs
        ItemID_dictionary_dataframe = pd.read_csv(filepath_or_buffer=movielens_tmdb_id_map_path, sep=",", header=0,
                                                  usecols=["movieId","imdbId","tmdbId"], dtype={0:str, 1:str, 2:str})

        ItemID_dictionary_tmdb_to_movielens = ItemID_dictionary_dataframe.set_index("tmdbId").to_dict()["movieId"]

        ICM_credits_dataframe["ItemID"] = [ItemID_dictionary_tmdb_to_movielens[itemID] for itemID in ICM_credits_dataframe["ItemID"]]
        ICM_metadata_dataframe["ItemID"] = [ItemID_dictionary_tmdb_to_movielens[itemID] for itemID in ICM_metadata_dataframe["ItemID"]]

        ICM_all_dataframe = pd.concat([ICM_credits_dataframe, ICM_metadata_dataframe])

        self._print("Loading Interactions")
        URM_all_dataframe, URM_timestamp_dataframe = _loadURM(URM_path, header=0, separator=',')

        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_all_dataframe, "URM_all")
        dataset_manager.add_ICM(ICM_credits_dataframe, "ICM_credits")
        dataset_manager.add_ICM(ICM_metadata_dataframe, "ICM_metadata")
        dataset_manager.add_ICM(ICM_all_dataframe, "ICM_all")

        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)



        self._print("Cleaning Temporary Files")

        shutil.rmtree(decompressed_zip_file_folder + "decompressed/", ignore_errors=True)

        self._print("Loading Complete")

        return loaded_dataset



    def _loadICM_credits(self, credits_path, header=True):

        credits_file = open(credits_path, 'r', encoding="utf8")

        if header:
            credits_file.readline()

        parser_credits = csv.reader(credits_file, delimiter=',', quotechar='"')

        movie_id_list = []
        cast_id_lists = []


        for newCredits in parser_credits:

            # newCredits is a tuple of two strings, both are lists of dictionaries
            # {'cast_id': 14, 'character': 'Woody (voice)', 'credit_id': '52fe4284c3a36847f8024f95', 'gender': 2, 'id': 31, 'name': 'Tom Hanks', 'order': 0, 'profile_path': '/pQFoyx7rp09CJTAb932F2g8Nlho.jpg'}
            # {'cast_id': 14, 'character': 'Woody (voice)', 'credit_id': '52fe4284c3a36847f8024f95', 'gender': 2, 'id': 31, 'name': 'Tom Hanks', 'order': 0, 'profile_path': '/pQFoyx7rp09CJTAb932F2g8Nlho.jpg'}
            # NOTE: sometimes a dict value is ""Savannah 'Vannah' Jackson"", if the previous eval removes the commas "" "" then the parsing of the string will fail
            cast_list = []
            credits_list = []

            try:
                cast_list = ast.literal_eval(newCredits[0])
                credits_list = ast.literal_eval(newCredits[1])
            except Exception as e:
                print("TheMoviesDatasetReader: Exception while parsing: '{}', skipping".format(str(e)))


            movie_id_list.append(newCredits[2])
            cast_list.extend(credits_list)

            this_cast_list = [cast_member["name"] for cast_member in cast_list]
            cast_id_lists.append(this_cast_list)


        ICM_dataframe = pd.DataFrame(cast_id_lists, index=movie_id_list).stack()
        ICM_dataframe = ICM_dataframe.reset_index()[["level_0", 0]]
        ICM_dataframe.columns = ['ItemID', 'FeatureID']
        ICM_dataframe["Data"] = 1

        return ICM_dataframe





    def _loadICM_metadata(self, metadata_path):


        # Some movies have newlines in the description before the end of the CSV
        # So I remove those new lines. F and T are the first possible letters for the first attribute column
        metadata_file = open(metadata_path, 'r', encoding="utf8")

        metadata_file_text = metadata_file.read()
        metadata_file_text = re.sub(r"\n([^FT])", r" \1", metadata_file_text)

        metadata_path = metadata_path + "_cleaned.txt"
        metadata_file.close()

        metadata_file_cleaned = open(metadata_path, 'w', encoding="utf8")
        metadata_file_cleaned.write(metadata_file_text)
        metadata_file_cleaned.close()


        columns = ["adult","belongs_to_collection","budget","genres","homepage","id",
                    "imdb_id","original_language","original_title","overview","popularity",
                    "poster_path","production_companies","production_countries",
                    "release_date","revenue","runtime","spoken_languages","status",
                    "tagline","title","video","vote_average","vote_count"]

        ICM_metadata_dataframe = pd.read_csv(filepath_or_buffer=metadata_path, sep=",", header=0,
                                    usecols = columns, dtype= {index:str for index in range(len(columns))})


        for column_json in ["belongs_to_collection", "genres", "production_companies", "production_countries", "spoken_languages"]:
            ICM_metadata_dataframe[column_json] = [ast.literal_eval(x) if not pd.isna(x) else x for x in ICM_metadata_dataframe[column_json]]

        movie_id_list = []
        token_lists = []

        for index, data_row in ICM_metadata_dataframe.iterrows():

            movie_id_list.append(data_row["id"])

            this_tokens = []
            this_tokens.append("adult_" + str(data_row["adult"]))

            if not pd.isna(data_row["belongs_to_collection"]):
                this_tokens.append("collection_" + str(data_row["belongs_to_collection"]["name"]))

            this_tokens.extend(["genre_" + str(x["name"]) for x in data_row["genres"]])
            this_tokens.append("original_language_" + str(data_row["original_language"]))
            this_tokens.extend(["production_companies_" + str(x["name"]) for x in data_row["production_companies"]])
            this_tokens.extend(["production_countries_" + str(x["iso_3166_1"]) for x in data_row["production_countries"]])
            this_tokens.extend(["spoken_languages_" + str(x["iso_639_1"]) for x in data_row["spoken_languages"]])

            token_lists.append(this_tokens)


        ICM_metadata_dataframe = pd.DataFrame(token_lists, index=movie_id_list).stack()
        ICM_metadata_dataframe = ICM_metadata_dataframe.reset_index()[["level_0", 0]]
        ICM_metadata_dataframe.columns = ['ItemID', 'FeatureID']
        ICM_metadata_dataframe["Data"] = 1

        return ICM_metadata_dataframe

