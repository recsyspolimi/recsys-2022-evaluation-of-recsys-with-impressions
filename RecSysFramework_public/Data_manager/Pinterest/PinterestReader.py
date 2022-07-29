#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Anonymous
"""

import pandas as pd
import zipfile, shutil
from Data_manager.DataReader import DataReader
from Data_manager.DatasetMapperManager import DatasetMapperManager
from Data_manager.DataReader_utils import download_from_URL, remove_Dataframe_duplicates

class PinterestReader(DataReader):
    """
    Geng, X., Zhang, H., Bian, J., & Chua, T. S. (2015). Learning image and user features for recommendation in social networks.
    In Proceedings of the IEEE International Conference on Computer Vision (pp. 4274-4282).

    https://openaccess.thecvf.com/content_iccv_2015/papers/Geng_Learning_Image_and_ICCV_2015_paper.pdf

    From README:
        This dataset a subset of our experimental dataset from mongodb and is only for research use.

        subset_iccv_board_pin.bson: includes pins of different boards. Each board represents a user.
        subset_iccv_pin_im.bson: includes pin_image pairs. Different pins may refer to the same images.  Each image has url which can be downloaded.
        subset_iccv_board_cate.bson: includes the groundtruths of boards' categories.
        categories.txt: 468 categories. the index of each category in the db is the line number of categories.txt

        If you have any futher question, please email to snownus@gmail.com


    Unfortunately the original data could not be parsed correctly. Hence this reader uses a preprocessed version of the dataset
    as done in the paper
    He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017, April). Neural collaborative filtering.
    In Proceedings of the 26th international conference on world wide web (pp. 173-182).

    From the Neural collaborative filtering paper:
        The original data is very large but highly sparse. For example, over 20% of users have only one pin, making it difficult
        to evaluate collaborative filtering algorithms. As such, we filtered the dataset in the same way as the MovieLens data
        that retained only users with at least 20 interactions (pins). This results in a subset of the data that contains 55, 187
        users and 1, 500, 809 interactions. Each interaction denotes whether the user has pinned the image to her own board.

    """

    DATASET_URL = "https://github.com/hexiangnan/neural_collaborative_filtering/archive/refs/heads/master.zip"
    # DATASET_URL = "https://sites.google.com/site/xueatalphabeta/academic-projects"
    DATASET_SUBFOLDER = "Pinterest/"
    AVAILABLE_URM = ["URM_all", "URM_occurrence"]
    AVAILABLE_ICM = []
    AVAILABLE_UCM = []
    DATASET_SPECIFIC_MAPPER = []

    IS_IMPLICIT = True

    def __init__(self):
        super(PinterestReader, self).__init__()


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        self._print("Loading original data")

        zipFile_path = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "neural_collaborative_filtering-master.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            self._print("Unable to find data zip file. Downloading...")

            download_from_URL(self.DATASET_URL, zipFile_path, "neural_collaborative_filtering-master.zip")

            dataFile = zipfile.ZipFile(zipFile_path + "neural_collaborative_filtering-master.zip")



        inner_path_in_zip = "neural_collaborative_filtering-master/Data/"


        URM_train_path = dataFile.extract(inner_path_in_zip + "pinterest-20.train.rating", path=zipFile_path + "decompressed/")
        URM_test_path = dataFile.extract(inner_path_in_zip + "pinterest-20.test.rating", path=zipFile_path + "decompressed/")

        dtype = {0:str, 1:str, 2:int, 3:int}

        URM_all_dataframe = pd.concat([pd.read_csv(URM_train_path, delimiter="\t", header = None, dtype = dtype),
                                       pd.read_csv(URM_test_path, delimiter="\t", header = None, dtype = dtype)], ignore_index=True)

        URM_all_dataframe.columns = ["UserID", "ItemID", "Data", "Zero"]
        URM_all_dataframe.drop(["Zero"], axis=1, inplace=True)
        URM_all_dataframe = URM_all_dataframe[URM_all_dataframe["Data"]>0.0]

        URM_occurrence_dataframe = URM_all_dataframe.groupby(["UserID","ItemID"],as_index=False)["Data"].sum()

        URM_all_dataframe = remove_Dataframe_duplicates(URM_all_dataframe,
                                                        unique_values_in_columns = ['UserID', 'ItemID'],
                                                        keep_highest_value_in_col = "Data")

        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_all_dataframe, "URM_all")
        dataset_manager.add_URM(URM_occurrence_dataframe, "URM_occurrence")

        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)



        self._print("Cleaning Temporary Files")

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        self._print("Loading Complete")

        return loaded_dataset






        #
        # compressed_file_folder = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER
        # decompressed_file_folder = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER
        #
        #
        # try:
        #
        #     dataFile = zipfile.ZipFile(compressed_file_folder + "pinterest_iccv.zip")
        #
        # except (FileNotFoundError, zipfile.BadZipFile):
        #
        #     self._print("Unable to find data zip file.")
        #     self._print("Automatic download not available, please ensure the compressed data file is in folder {}.".format(compressed_file_folder))
        #     self._print("Data can be downloaded here: {}".format(self.DATASET_URL))
        #
        #     # If directory does not exist, create
        #     if not os.path.exists(compressed_file_folder):
        #         os.makedirs(compressed_file_folder)
        #
        #     raise FileNotFoundError("Automatic download not available.")
        #
        #
        # dataFile.extractall(path=decompressed_file_folder + "decompressed/")
        # URM_all, self.item_original_ID_to_index, self.user_original_ID_to_index = self._load_pins(decompressed_file_folder + "decompressed/")
        #
        #
        # loaded_URM_dict = {"URM_all": URM_all}
        #
        # loaded_dataset = Dataset(dataset_name = self._get_dataset_name(),
        #                          URM_dictionary = loaded_URM_dict,
        #                          ICM_dictionary = None,
        #                          ICM_feature_mapper_dictionary = None,
        #                          UCM_dictionary = None,
        #                          UCM_feature_mapper_dictionary = None,
        #                          user_original_ID_to_index= self.user_original_ID_to_index,
        #                          item_original_ID_to_index= self.item_original_ID_to_index,
        #                          is_implicit = self.IS_IMPLICIT,
        #                          )
        #
        # self._print("Cleaning Temporary Files")
        #
        # shutil.rmtree(decompressed_file_folder + "decompressed", ignore_errors=True)
        #
        # self._print("Loading Complete")
        #
        # return loaded_dataset


    #
    #
    # def _loadURM(self, filename):
    #     '''
    #     Read .rating file and Return dok matrix.
    #     The first line of .rating file is: num_users\t num_items
    #     '''
    #
    #     # Get number of users and items
    #     import scipy.sparse as sps
    #
    #
    #     with open(filename, "r") as f:
    #         line = f.readline()
    #         while line != None and line != "":
    #             arr = line.split("\t")
    #             user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
    #             if (rating > 0):
    #                 mat[user, item] = 1.0
    #             line = f.readline()
    #     return mat
    #
    #


    #
    #
    #
    # def _load_pins(self, folder_path):
    #
    #     file = open(folder_path + 'pinterest_iccv/subset_iccv_pin_im.bson', "rb")
    #     #data = bson.decode_all(file.read())
    #     #data = bson.loads(file.read())
    #
    #     Load mapping pin_id to image_id
    #     pin_id_to_image_id = {}
    #
    #     for line in file:
    #
    #         try:
    #             data_row = bson.loads(line)
    #
    #             pin_id = data_row["pin_id"]
    #
    #             image_id = data_row["im_name"]
    #
    #             pin_id_to_image_id[pin_id] = image_id
    #         except:
    #             pass
    #
    #
    #
    #
    #
    #
    #     file = open(folder_path + 'pinterest_iccv/subset_iccv_board_pins.bson', "rb")
    #     #data = bson.decode_all(file.read())
    #
    #     # URM_pins = IncrementalSparseMatrix(auto_create_col_mapper=True, auto_create_row_mapper=True)
    #
    #     user_id_list = []
    #     user_pin_list = []
    #
    #     for line in file:
    #
    #         data_row = bson.loads(line)
    #
    #         user_id_list.append(data_row["board_id"])
    #
    #         pins_list = data_row["pins"]
    #         # image_id_list = [pin_id_to_image_id[pin_id] for pin_id in pins_list]
    #         user_pin_list.append(pins_list)
    #
    #         # URM_pins.add_single_row(user_id, image_id_list, data=1.0)
    #
    #     URM_dataframe = pd.DataFrame(user_pin_list, index=user_id_list).stack()
    #     URM_dataframe = URM_dataframe.reset_index()[["level_0", 0]]
    #     URM_dataframe.columns = ['UserID', 'ItemID']
    #     URM_dataframe["Data"] = 1
    #
    #     return URM_dataframe
    #






