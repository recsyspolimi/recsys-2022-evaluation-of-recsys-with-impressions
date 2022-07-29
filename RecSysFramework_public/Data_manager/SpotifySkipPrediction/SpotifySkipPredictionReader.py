#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Anonymous
"""

import numpy as np
import pandas as pd
import tarfile, os, shutil
import scipy.sparse as sps
from Data_manager.Dataset import Dataset
from Data_manager.DataReader import DataReader


class SpotifySkipPredictionReader(DataReader):

    DATASET_URL = "https://www.crowdai.org/challenges/spotify-sequential-skip-prediction-challenge"

    DATASET_SUBFOLDER = "SpotifySkipPrediction/"
    AVAILABLE_ICM = ["ICM_all"]



    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        self._print("Loading original data")

        compressed_file_folder = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER
        decompressed_file_folder = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER


        try:

            compressed_train_set_file = tarfile.open(compressed_file_folder + "20181113_training_set.tar.gz", "r:gz")


        except (FileNotFoundError, tarfile.ReadError, tarfile.ExtractError):

            self._print("Unable to fild data zip file.")
            self._print("Automatic download not available, please ensure the compressed data file is in folder {}.".format(compressed_file_folder))
            self._print("Data can be downloaded here: {}".format(self.DATASET_URL))

            # If directory does not exist, create
            if not os.path.exists(compressed_file_folder):
                os.makedirs(compressed_file_folder)

            raise FileNotFoundError("Automatic download not available.")


        #session_id,session_position,session_length,track_id_clean,skip_1,skip_2,skip_3,not_skipped,context_switch,no_pause_before_play,short_pause_before_play,long_pause_before_play,hist_user_behavior_n_seekfwd,hist_user_behavior_n_seekback,hist_user_behavior_is_shuffle,hour_of_day,date,premium,context_type,hist_user_behavior_reason_start,hist_user_behavior_reason_end



        from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

        URM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = self.item_original_ID_to_index, on_new_col = "add",
                                                        preinitialized_row_mapper = None, on_new_row = "add", dtype = np.bool)



        #print("Number of files is {}".format(len(compressed_train_set_file.getnames())))

        # If directory does not exist, create

        sps_blocks_path = decompressed_file_folder + "sps_blocks/"
        if not os.path.exists(sps_blocks_path):
            os.makedirs(sps_blocks_path)


        next_file = ""
        file_counter = 0
        interaction_counter = 0

        while next_file is not None:

            next_file = compressed_train_set_file.next()

            if file_counter<=650:
                if next_file.isfile():
                    file_counter+=1
                    print("Skipping file {}: '{}'".format(file_counter, next_file.path))
                continue

            if next_file is not None and next_file.isfile():

                print("Extracting: '{}'".format(next_file.path))

                compressed_train_set_file.extractall(path=decompressed_file_folder + "decompressed/", members=[next_file])

                decompressed_file_path = decompressed_file_folder + "decompressed/" + next_file.path

                self._load_URM_events(URM_builder, decompressed_file_path)

                file_counter+=1

                print("Loaded {}/660 files, {:.2E} interactions".format(file_counter, interaction_counter + URM_builder.get_nnz()))

                os.remove(decompressed_file_path)


            if file_counter % 50 == 0 or next_file is None:

                URM_all = URM_builder.get_SparseMatrix()

                print("Saving {}".format(sps_blocks_path + "URM_file_{}".format(file_counter)))

                sps.save_npz(sps_blocks_path + "URM_file_{}".format(file_counter), URM_all)

                self.item_original_ID_to_index = URM_builder.get_row_token_to_id_mapper()
                self.user_original_ID_to_index = URM_builder.get_column_token_to_id_mapper()

                interaction_counter += URM_builder.get_nnz()


                URM_builder = IncrementalSparseMatrix_FilterIDs(
                                preinitialized_col_mapper = self.item_original_ID_to_index,
                                on_new_col = "add",
                                preinitialized_row_mapper = self.user_original_ID_to_index,
                                on_new_row = "add",
                                dtype = np.bool)




        compressed_train_set_file.close()
        #
        #
        #
        # print("ThirtyMusicReader: loading ICM_tracks")
        # self.ICM_tracks, self.tokenToFeatureMapper_ICM_tracks, self.item_original_ID_to_index = self._load_ICM_tracks(tracks_path, if_new_item = "add")
        #


        print("Cleaning Temporary Files")

        shutil.rmtree(decompressed_file_folder + "decompressed/", ignore_errors=True)

        print("loading complete")








    def _load_ICM_tracks(self, tracks_path, if_new_item = "add"):

        from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

        ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = None, on_new_col = "add",
                                                        preinitialized_row_mapper = None, on_new_row = if_new_item)


        fileHandle = open(tracks_path, "r")


        numCells = 0

        for line in fileHandle:

            line = line.split("\t")

            #line[0]
            track_id = line[1]

            #line[2]
            line[3] = line[3].replace(':null', ':"null"')
            line[3] = line[3].replace(': null', ':"null"')
            line[3] = eval(line[3])


            track_duration = line[3]["duration"]
            track_playcount = line[3]["playcount"]
            track_MBID = line[3]["MBID"]
            track_name = line[3]["name"]

            line[4] = line[4].replace(':null', ':"null"')
            line[4] = eval(line[4])

            track_artists_list = line[4]["artists"]
            track_albums_list = line[4]["albums"]
            track_tags_list = line[4]["tags"]

            if track_albums_list == "null":
                track_albums_list = []
            if track_tags_list == "null":
                track_tags_list = []


            token_list = [*track_artists_list, *track_albums_list, *track_tags_list]

            token_list = ["{}_{}".format(new_token["type"], new_token["id"]) for new_token in token_list]

            ICM_builder.add_single_row(track_id, token_list, data=1.0)

            numCells += 1

            if numCells % 100000 == 0 and numCells!=0:
                print("Processed {} tracks".format(numCells))


        fileHandle.close()

        return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()




    #
    #
    # def _load_sessions(self, sessions_path, if_new_user = "add", if_new_item = "ignore"):
    #
    #
    #
    #     if if_new_user not in ["add", "ignore", "exception"]:
    #         raise ValueError("DataReader: if_new_user parameter not recognized. Accepted values are 'add', 'ignore', 'exception', provided was '{}'".format(if_new_user))
    #
    #     if if_new_item not in ["add", "ignore", "exception"]:
    #         raise ValueError("DataReader: if_new_item parameter not recognized. Accepted values are 'add', 'ignore', 'exception', provided was '{}'".format(if_new_item))
    #
    #     if if_new_user == "ignore":
    #         if_new_user_get_user_index = "exception"
    #     else:
    #         if_new_user_get_user_index = if_new_user
    #
    #     if if_new_item == "ignore":
    #         if_new_item_get_item_index = "exception"
    #     else:
    #         if_new_item_get_item_index = if_new_item
    #
    #
    #
    #
    #
    #     fileHandle = open(sessions_path, "r")
    #
    #     # Use array as for datasets this big lists would require more than 10GB of RAM
    #     dataBlock = 10000000
    #
    #     values = np.zeros(dataBlock, dtype=np.float64)
    #     rows = np.zeros(dataBlock, dtype=np.int32)
    #     cols = np.zeros(dataBlock, dtype=np.int32)
    #
    #     numCells = 0
    #
    #
    #
    #     for line in fileHandle:
    #
    #         line = line.split("\t")
    #
    #         #line[0]
    #         session_id = line[1]
    #         #line[2]
    #
    #         content = line[3].split(" ")
    #         content[0] = eval(content[0])
    #
    #         num_tracks = content[0]["numtracks"]
    #         play_time = content[0]["playtime"]
    #
    #         content[1] = content[1].replace(':null', ':"null"')
    #
    #         content[1] = eval(content[1])
    #
    #         subjects = content[1]["subjects"]
    #         type = subjects[0]["type"]
    #         user_id = subjects[0]["id"]
    #         user_index = self._get_user_index(user_id, if_new = if_new_user_get_user_index)
    #
    #         # Objects contains tracks
    #         objects = content[1]["objects"]
    #
    #         for new_track in objects:
    #
    #             if numCells % 1000000 == 0 and numCells!=0:
    #                 print("Processed {} rows".format(numCells))
    #
    #
    #             track_type = new_track["type"]
    #             track_id = new_track["id"]
    #
    #             track_playstart = new_track["playstart"]
    #             track_playtime = new_track["playtime"]
    #             track_playratio = new_track["playratio"]
    #
    #             track_action = new_track["action"]
    #
    #             try:
    #                 track_index = self._get_item_index(track_id, if_new = if_new_item_get_item_index)
    #             except KeyError:
    #                 # Go to next line
    #                 print("ThirtyMusicReader: Sessions contains track which is not in ICM: {}. Skipping...".format(track_id))
    #                 continue
    #
    #
    #             if numCells == len(rows):
    #                 rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
    #                 cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
    #                 values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float64)))
    #
    #             rows[numCells] = int(user_index)
    #             cols[numCells] = int(track_index)
    #             values[numCells] = True
    #
    #             numCells += 1
    #
    #
    #     fileHandle.close()
    #
    #     return sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])), dtype=np.float32)
    #






    def _load_URM_events(self, URM_builder, events_path):


        # session_id,session_position,session_length,track_id_clean,skip_1,skip_2,skip_3,not_skipped,context_switch,no_pause_before_play,short_pause_before_play,long_pause_before_play,hist_user_behavior_n_seekfwd,hist_user_behavior_n_seekback,hist_user_behavior_is_shuffle,hour_of_day,date,premium,context_type,hist_user_behavior_reason_start,hist_user_behavior_reason_end
        # 31_0000b0c5-94b8-426b-87e2-ef81510b9b17,1,20,t_86abc9b1-2a71-41d8-ab97-ac97ea20276a,true,true,true,false,0,0,0,0,0,0,true,8,2018-08-14,true,user_collection,fwdbtn,fwdbtn
        # 31_0000b0c5-94b8-426b-87e2-ef81510b9b17,2,20,t_33a133e6-240c-467d-a5c5-a6729a545cc2,true,true,true,false,0,0,1,1,0,0,true,8,2018-08-14,true,user_collection,fwdbtn,fwdbtn
        # 31_0000b0c5-94b8-426b-87e2-ef81510b9b17,3,20,t_cd87b117-d9d0-4562-b469-65ae0e88f8f5,true,true,true,false,0,1,0,0,0,0,true,8,2018-08-14,true,user_collection,fwdbtn,fwdbtn
        # 31_0000b0c5-94b8-426b-87e2-ef81510b9b17,4,20,t_de6bfde1-10b3-4984-add7-b41050bc9353,true,true,true,false,0,1,0,0,0,0,true,8,2018-08-14,true,user_collection,fwdbtn,fwdbtn
        # 31_0000b0c5-94b8-426b-87e2-ef81510b9b17,5,20,t_01d7104d-d28c-4c56-9012-d22ef2b8bdc9,false,false,false,true,0,1,0,0,0,0,true,8,2018-08-14,true,user_collection,fwdbtn,trackdone
        # 31_0000b0c5-94b8-426b-87e2-ef81510b9b17,6,20,t_ff674955-20ad-48bf-8494-d5fbe9dd7fac,false,false,true,false,0,0,1,1,0,0,true,8,2018-08-14,true,user_collection,trackdone,fwdbtn

        # session_id,                           31_0000b0c5-94b8-426b-87e2-ef81510b9b17
        # session_position,                     1
        # session_length,                       20
        # track_id_clean,                       t_86abc9b1-2a71-41d8-ab97-ac97ea20276a
        # skip_1,                               true
        # skip_2,                               true
        # skip_3,                               true
        # not_skipped,                          false
        # context_switch,                       0
        # no_pause_before_play,                 0
        # short_pause_before_play,              0
        # long_pause_before_play,               0
        # hist_user_behavior_n_seekfwd,         0
        # hist_user_behavior_n_seekback,        0
        # hist_user_behavior_is_shuffle,        true
        # hour_of_day,                          8
        # date,                                 2018-08-14
        # premium,                              true
        # context_type,                         user_collection
        # hist_user_behavior_reason_start,      fwdbtn
        # hist_user_behavior_reason_end         fwdbtn




        data_frame = pd.read_csv(events_path)


        session_id_list = data_frame["session_id"].values
        track_id_list = data_frame["track_id_clean"].values

        not_skipped_list = data_frame["not_skipped"].values
        skipped_list = np.logical_not(not_skipped_list)



        URM_builder.add_data_lists(session_id_list, track_id_list, skipped_list)


        # numCells = 0
        #
        # for index, row in data_frame.iterrows():
        #
        #     URM_builder.add_single_row(row.session_id, row.track_id_clean, data=1.0)
        #
        #     if numCells % 500000 == 0:
        #         print("Processed {} rows".format(numCells))
        #
        #     numCells += 1
        #











