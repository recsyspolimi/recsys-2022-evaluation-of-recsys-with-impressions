#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/01/18

@author: Anonymous
"""

import zipfile, os, csv, shutil
import numpy as np
import scipy.sparse as sps
from Data_manager.Dataset import Dataset
from Data_manager.DataReader import DataReader
from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs
from Data_manager.Movielens.Movielens20MReader import Movielens20MReader
from Data_manager.DataReader_utils import reconcile_mapper_with_removed_tokens


class MultifacetedMovieTrailerFeatureReader(DataReader):

    DATASET_URL = ["https://mmprj.github.io/mtrm_dataset/index",
                   "https://polimi365-my.sharepoint.com/:u:/g/personal/10322330_polimi_it/EU1QXiU0jJFFsf-7zf-gIMwBbYy7AUegDFqfwnNncW2e-g?e=h8dDrT"
                   ]

    DATASET_SUBFOLDER = "MultifacetedMovieTrailerFeature/"
    DATASET_SPECIFIC_MAPPER = []

    IS_IMPLICIT = False

    AVAILABLE_ICM = [
            "ICM_Aesthetic_AvgVar_All",
            # "ICM_Aesthetic_AvgVar_Feat26Brightness",
            # "ICM_Aesthetic_AvgVar_Feat26BrightnessSegments",
            # "ICM_Aesthetic_AvgVar_Feat26Centroids",
            # "ICM_Aesthetic_AvgVar_Feat26ColorModel",
            # "ICM_Aesthetic_AvgVar_Feat26Colorfulness",
            # "ICM_Aesthetic_AvgVar_Feat26ContrastSegments",
            # "ICM_Aesthetic_AvgVar_Feat26Convexity",
            # "ICM_Aesthetic_AvgVar_Feat26Coordinates",
            # "ICM_Aesthetic_AvgVar_Feat26DofIndicator",
            # "ICM_Aesthetic_AvgVar_Feat26Edge",
            # "ICM_Aesthetic_AvgVar_Feat26HSL",
            # "ICM_Aesthetic_AvgVar_Feat26HSV",
            # "ICM_Aesthetic_AvgVar_Feat26HueDescriptors",
            # "ICM_Aesthetic_AvgVar_Feat26HueModels",
            # "ICM_Aesthetic_AvgVar_Feat26HueSegments",
            # "ICM_Aesthetic_AvgVar_Feat26LargeSegments",
            # "ICM_Aesthetic_AvgVar_Feat26MassVarianceSegments",
            # "ICM_Aesthetic_AvgVar_Feat26RGBEntropy",
            # "ICM_Aesthetic_AvgVar_Feat26SaturationSegments",
            # "ICM_Aesthetic_AvgVar_Feat26SkewnessSegments",
            # "ICM_Aesthetic_AvgVar_Feat26Texture",
            # "ICM_Aesthetic_AvgVar_Feat26ValueSegments",
            # "ICM_Aesthetic_AvgVar_Feat26WaveletHSV",
            # "ICM_Aesthetic_AvgVar_Feat26avgHSLFocus",
            # "ICM_Aesthetic_AvgVar_Feat26avgHSVRot",
            # "ICM_Aesthetic_AvgVar_Feat26avgWaveletHSV",
            # "ICM_Aesthetic_AvgVar_Type3Color",
            # "ICM_Aesthetic_AvgVar_Type3Object",
            # "ICM_Aesthetic_AvgVar_Type3Texture",
            "ICM_Aesthetic_Avg_All",
            # "ICM_Aesthetic_Avg_Feat26Brightness",
            # "ICM_Aesthetic_Avg_Feat26BrightnessSegments",
            # "ICM_Aesthetic_Avg_Feat26Centroids",
            # "ICM_Aesthetic_Avg_Feat26ColorModel",
            # "ICM_Aesthetic_Avg_Feat26Colorfulness",
            # "ICM_Aesthetic_Avg_Feat26ContrastSegments",
            # "ICM_Aesthetic_Avg_Feat26Convexity",
            # "ICM_Aesthetic_Avg_Feat26Coordinates",
            # "ICM_Aesthetic_Avg_Feat26DofIndicator",
            # "ICM_Aesthetic_Avg_Feat26Edge",
            # "ICM_Aesthetic_Avg_Feat26HSL",
            # "ICM_Aesthetic_Avg_Feat26HSV",
            # "ICM_Aesthetic_Avg_Feat26HueDescriptors",
            # "ICM_Aesthetic_Avg_Feat26HueModels",
            # "ICM_Aesthetic_Avg_Feat26HueSegments",
            # "ICM_Aesthetic_Avg_Feat26LargeSegments",
            # "ICM_Aesthetic_Avg_Feat26MassVarianceSegments",
            # "ICM_Aesthetic_Avg_Feat26RGBEntropy",
            # "ICM_Aesthetic_Avg_Feat26SaturationSegments",
            # "ICM_Aesthetic_Avg_Feat26SkewnessSegments",
            # "ICM_Aesthetic_Avg_Feat26Texture",
            # "ICM_Aesthetic_Avg_Feat26ValueSegments",
            # "ICM_Aesthetic_Avg_Feat26WaveletHSV",
            # "ICM_Aesthetic_Avg_Feat26avgHSLFocus",
            # "ICM_Aesthetic_Avg_Feat26avgHSVRot",
            # "ICM_Aesthetic_Avg_Feat26avgWaveletHSV",
            # "ICM_Aesthetic_Avg_Type3Color",
            # "ICM_Aesthetic_Avg_Type3Object",
            # "ICM_Aesthetic_Avg_Type3Texture",
            "ICM_Aesthetic_MedMad_All",
            # "ICM_Aesthetic_MedMad_Feat26Brightness",
            # "ICM_Aesthetic_MedMad_Feat26BrightnessSegments",
            # "ICM_Aesthetic_MedMad_Feat26Centroids",
            # "ICM_Aesthetic_MedMad_Feat26ColorModel",
            # "ICM_Aesthetic_MedMad_Feat26Colorfulness",
            # "ICM_Aesthetic_MedMad_Feat26ContrastSegments",
            # "ICM_Aesthetic_MedMad_Feat26Convexity",
            # "ICM_Aesthetic_MedMad_Feat26Coordinates",
            # "ICM_Aesthetic_MedMad_Feat26DofIndicator",
            # "ICM_Aesthetic_MedMad_Feat26Edge",
            # "ICM_Aesthetic_MedMad_Feat26HSL",
            # "ICM_Aesthetic_MedMad_Feat26HSV",
            # "ICM_Aesthetic_MedMad_Feat26HueDescriptors",
            # "ICM_Aesthetic_MedMad_Feat26HueModels",
            # "ICM_Aesthetic_MedMad_Feat26HueSegments",
            # "ICM_Aesthetic_MedMad_Feat26LargeSegments",
            # "ICM_Aesthetic_MedMad_Feat26MassVarianceSegments",
            # "ICM_Aesthetic_MedMad_Feat26RGBEntropy",
            # "ICM_Aesthetic_MedMad_Feat26SaturationSegments",
            # "ICM_Aesthetic_MedMad_Feat26SkewnessSegments",
            # "ICM_Aesthetic_MedMad_Feat26Texture",
            # "ICM_Aesthetic_MedMad_Feat26ValueSegments",
            # "ICM_Aesthetic_MedMad_Feat26WaveletHSV",
            # "ICM_Aesthetic_MedMad_Feat26avgHSLFocus",
            # "ICM_Aesthetic_MedMad_Feat26avgHSVRot",
            # "ICM_Aesthetic_MedMad_Feat26avgWaveletHSV",
            # "ICM_Aesthetic_MedMad_Type3Color",
            # "ICM_Aesthetic_MedMad_Type3Object",
            # "ICM_Aesthetic_MedMad_Type3Texture",
            "ICM_Aesthetic_Med_All",
            # "ICM_Aesthetic_Med_Feat26Brightness",
            # "ICM_Aesthetic_Med_Feat26BrightnessSegments",
            # "ICM_Aesthetic_Med_Feat26Centroids",
            # "ICM_Aesthetic_Med_Feat26ColorModel",
            # "ICM_Aesthetic_Med_Feat26Colorfulness",
            # "ICM_Aesthetic_Med_Feat26ContrastSegments",
            # "ICM_Aesthetic_Med_Feat26Convexity",
            # "ICM_Aesthetic_Med_Feat26Coordinates",
            # "ICM_Aesthetic_Med_Feat26DofIndicator",
            # "ICM_Aesthetic_Med_Feat26Edge",
            # "ICM_Aesthetic_Med_Feat26HSL",
            # "ICM_Aesthetic_Med_Feat26HSV",
            # "ICM_Aesthetic_Med_Feat26HueDescriptors",
            # "ICM_Aesthetic_Med_Feat26HueModels",
            # "ICM_Aesthetic_Med_Feat26HueSegments",
            # "ICM_Aesthetic_Med_Feat26LargeSegments",
            # "ICM_Aesthetic_Med_Feat26MassVarianceSegments",
            # "ICM_Aesthetic_Med_Feat26RGBEntropy",
            # "ICM_Aesthetic_Med_Feat26SaturationSegments",
            # "ICM_Aesthetic_Med_Feat26SkewnessSegments",
            # "ICM_Aesthetic_Med_Feat26Texture",
            # "ICM_Aesthetic_Med_Feat26ValueSegments",
            # "ICM_Aesthetic_Med_Feat26WaveletHSV",
            # "ICM_Aesthetic_Med_Feat26avgHSLFocus",
            # "ICM_Aesthetic_Med_Feat26avgHSVRot",
            # "ICM_Aesthetic_Med_Feat26avgWaveletHSV",
            # "ICM_Aesthetic_Med_Type3Color",
            # "ICM_Aesthetic_Med_Type3Object",
            # "ICM_Aesthetic_Med_Type3Texture",
            "ICM_AlexNet_Avg",
            "ICM_AlexNet_AvgVar",
            "ICM_AlexNet_Med",
            "ICM_AlexNet_MedMad",
            "ICM_BLF_Correlation",
            "ICM_BLF_DeltaSpectral",
            "ICM_BLF_LogarithmicFluctuation",
            "ICM_BLF_Spectral",
            "ICM_BLF_SpectralContrast",
            "ICM_BLF_VarianceDeltaSpectral",
            "ICM_Genre",
            "ICM_Ivector_gmm_128_tvDim_10",
            "ICM_Ivector_gmm_128_tvDim_100",
            "ICM_Ivector_gmm_128_tvDim_20",
            "ICM_Ivector_gmm_128_tvDim_200",
            "ICM_Ivector_gmm_128_tvDim_40",
            "ICM_Ivector_gmm_128_tvDim_400",
            "ICM_Ivector_gmm_16_tvDim_10",
            "ICM_Ivector_gmm_16_tvDim_100",
            "ICM_Ivector_gmm_16_tvDim_20",
            "ICM_Ivector_gmm_16_tvDim_200",
            "ICM_Ivector_gmm_16_tvDim_40",
            "ICM_Ivector_gmm_16_tvDim_400",
            "ICM_Ivector_gmm_256_tvDim_10",
            "ICM_Ivector_gmm_256_tvDim_100",
            "ICM_Ivector_gmm_256_tvDim_20",
            "ICM_Ivector_gmm_256_tvDim_200",
            "ICM_Ivector_gmm_256_tvDim_40",
            "ICM_Ivector_gmm_256_tvDim_400",
            "ICM_Ivector_gmm_32_tvDim_10",
            "ICM_Ivector_gmm_32_tvDim_100",
            "ICM_Ivector_gmm_32_tvDim_20",
            "ICM_Ivector_gmm_32_tvDim_200",
            "ICM_Ivector_gmm_32_tvDim_40",
            "ICM_Ivector_gmm_32_tvDim_400",
            "ICM_Ivector_gmm_512_tvDim_10",
            "ICM_Ivector_gmm_512_tvDim_100",
            "ICM_Ivector_gmm_512_tvDim_20",
            "ICM_Ivector_gmm_512_tvDim_200",
            "ICM_Ivector_gmm_512_tvDim_40",
            "ICM_Ivector_gmm_512_tvDim_400",
            "ICM_Ivector_gmm_64_tvDim_10",
            "ICM_Ivector_gmm_64_tvDim_100",
            "ICM_Ivector_gmm_64_tvDim_20",
            "ICM_Ivector_gmm_64_tvDim_200",
            "ICM_Ivector_gmm_64_tvDim_40",
            "ICM_Ivector_gmm_64_tvDim_400",
            "ICM_Tag",
            "ICM_Year"
             ]

    # AVAILABLE_ICM = [
    #             "ICM_Tag",
    #             "ICM_Genre",
    #             "ICM_Year",
    #             "ICM_BLF_Correlation",
    #             "ICM_BLF_DeltaSpectral",
    #             "ICM_BLF_LogarithmicFluctuation",
    #             "ICM_BLF_SpectralContrast",
    #             "ICM_BLF_Spectral",
    #             "ICM_BLF_VarianceDeltaSpectral",
    #             "ICM_Ivector_gmm_256_tvDim_40",
    #             "ICM_Ivector_gmm_256_tvDim_100",
    #             "ICM_Ivector_gmm_256_tvDim_200",
    #             "ICM_Ivector_gmm_256_tvDim_400",
    #             "ICM_Ivector_gmm_512_tvDim_40",
    #             "ICM_Ivector_gmm_512_tvDim_100",
    #             "ICM_Ivector_gmm_512_tvDim_200",
    #             "ICM_Ivector_gmm_512_tvDim_400",
    #             "ICM_AlexNet_Avg",
    #             "ICM_Aesthetic_Avg",
    #             "ICM_AlexNet_AvgVar",
    #             "ICM_Aesthetic_AvgVar",
    #             "ICM_AlexNet_Med",
    #             "ICM_Aesthetic_Med",
    #             "ICM_AlexNet_MedMad",
    #             "ICM_Aesthetic_MedMad",
    #             ]



    # AVAILABLE_ICM = [
    #         # "Audio/Block level features/All/blf_sim_matrix",
    #         # "Audio/Block level features/All/movieIds_sim",
    #         "Audio/Block level features/Component6/BLF_CORRELATIONfeat",
    #         "Audio/Block level features/Component6/BLF_DELTASPECTRALfeat",
    #         "Audio/Block level features/Component6/BLF_LOGARITHMICFLUCTUATIONfeat",
    #         "Audio/Block level features/Component6/BLF_SPECTRALCONTRASTfeat",
    #         "Audio/Block level features/Component6/BLF_SPECTRALfeat",
    #         "Audio/Block level features/Component6/BLF_VARIANCEDELTASPECTRALfeat",
    #         "Audio/ivector features/IVec_splitItem_fold_1_gmm_128_tvDim_10",
    #         "Audio/ivector features/IVec_splitItem_fold_1_gmm_128_tvDim_100",
    #         "Audio/ivector features/IVec_splitItem_fold_1_gmm_128_tvDim_20",
    #         "Audio/ivector features/IVec_splitItem_fold_1_gmm_128_tvDim_200",
    #         "Audio/ivector features/IVec_splitItem_fold_1_gmm_128_tvDim_40",
    #         "Audio/ivector features/IVec_splitItem_fold_1_gmm_128_tvDim_400",
    #         "Audio/ivector features/IVec_splitItem_fold_1_gmm_16_tvDim_10",
    #         "Audio/ivector features/IVec_splitItem_fold_1_gmm_16_tvDim_100",
    #         "Audio/ivector features/IVec_splitItem_fold_1_gmm_16_tvDim_20",
    #         "Audio/ivector features/IVec_splitItem_fold_1_gmm_16_tvDim_200",
    #         "Audio/ivector features/IVec_splitItem_fold_1_gmm_16_tvDim_40",
    #         "Audio/ivector features/IVec_splitItem_fold_1_gmm_16_tvDim_400",
    #         "Audio/ivector features/IVec_splitItem_fold_1_gmm_256_tvDim_10",
    #         "Audio/ivector features/IVec_splitItem_fold_1_gmm_256_tvDim_100",
    #         "Audio/ivector features/IVec_splitItem_fold_1_gmm_256_tvDim_20",
    #         "Audio/ivector features/IVec_splitItem_fold_1_gmm_256_tvDim_200",
    #         "Audio/ivector features/IVec_splitItem_fold_1_gmm_256_tvDim_40",
    #         "Audio/ivector features/IVec_splitItem_fold_1_gmm_256_tvDim_400",
    #         "Audio/ivector features/IVec_splitItem_fold_1_gmm_32_tvDim_10",
    #         "Audio/ivector features/IVec_splitItem_fold_1_gmm_32_tvDim_100",
    #         "Audio/ivector features/IVec_splitItem_fold_1_gmm_32_tvDim_20",
    #         "Audio/ivector features/IVec_splitItem_fold_1_gmm_32_tvDim_200",
    #         "Audio/ivector features/IVec_splitItem_fold_1_gmm_32_tvDim_40",
    #         "Audio/ivector features/IVec_splitItem_fold_1_gmm_32_tvDim_400",
    #         "Audio/ivector features/IVec_splitItem_fold_1_gmm_512_tvDim_10",
    #         "Audio/ivector features/IVec_splitItem_fold_1_gmm_512_tvDim_100",
    #         "Audio/ivector features/IVec_splitItem_fold_1_gmm_512_tvDim_20",
    #         "Audio/ivector features/IVec_splitItem_fold_1_gmm_512_tvDim_200",
    #         "Audio/ivector features/IVec_splitItem_fold_1_gmm_512_tvDim_40",
    #         "Audio/ivector features/IVec_splitItem_fold_1_gmm_512_tvDim_400",
    #         "Audio/ivector features/IVec_splitItem_fold_1_gmm_64_tvDim_10",
    #         "Audio/ivector features/IVec_splitItem_fold_1_gmm_64_tvDim_100",
    #         "Audio/ivector features/IVec_splitItem_fold_1_gmm_64_tvDim_20",
    #         "Audio/ivector features/IVec_splitItem_fold_1_gmm_64_tvDim_200",
    #         "Audio/ivector features/IVec_splitItem_fold_1_gmm_64_tvDim_40",
    #         "Audio/ivector features/IVec_splitItem_fold_1_gmm_64_tvDim_400",
    #         "Audio/ivector features/IVec_splitItem_fold_2_gmm_128_tvDim_10",
    #         "Audio/ivector features/IVec_splitItem_fold_2_gmm_128_tvDim_100",
    #         "Audio/ivector features/IVec_splitItem_fold_2_gmm_128_tvDim_20",
    #         "Audio/ivector features/IVec_splitItem_fold_2_gmm_128_tvDim_200",
    #         "Audio/ivector features/IVec_splitItem_fold_2_gmm_128_tvDim_40",
    #         "Audio/ivector features/IVec_splitItem_fold_2_gmm_128_tvDim_400",
    #         "Audio/ivector features/IVec_splitItem_fold_2_gmm_16_tvDim_10",
    #         "Audio/ivector features/IVec_splitItem_fold_2_gmm_16_tvDim_100",
    #         "Audio/ivector features/IVec_splitItem_fold_2_gmm_16_tvDim_20",
    #         "Audio/ivector features/IVec_splitItem_fold_2_gmm_16_tvDim_200",
    #         "Audio/ivector features/IVec_splitItem_fold_2_gmm_16_tvDim_40",
    #         "Audio/ivector features/IVec_splitItem_fold_2_gmm_16_tvDim_400",
    #         "Audio/ivector features/IVec_splitItem_fold_2_gmm_256_tvDim_10",
    #         "Audio/ivector features/IVec_splitItem_fold_2_gmm_256_tvDim_100",
    #         "Audio/ivector features/IVec_splitItem_fold_2_gmm_256_tvDim_20",
    #         "Audio/ivector features/IVec_splitItem_fold_2_gmm_256_tvDim_200",
    #         "Audio/ivector features/IVec_splitItem_fold_2_gmm_256_tvDim_40",
    #         "Audio/ivector features/IVec_splitItem_fold_2_gmm_256_tvDim_400",
    #         "Audio/ivector features/IVec_splitItem_fold_2_gmm_32_tvDim_10",
    #         "Audio/ivector features/IVec_splitItem_fold_2_gmm_32_tvDim_100",
    #         "Audio/ivector features/IVec_splitItem_fold_2_gmm_32_tvDim_20",
    #         "Audio/ivector features/IVec_splitItem_fold_2_gmm_32_tvDim_200",
    #         "Audio/ivector features/IVec_splitItem_fold_2_gmm_32_tvDim_40",
    #         "Audio/ivector features/IVec_splitItem_fold_2_gmm_32_tvDim_400",
    #         "Audio/ivector features/IVec_splitItem_fold_2_gmm_512_tvDim_10",
    #         "Audio/ivector features/IVec_splitItem_fold_2_gmm_512_tvDim_100",
    #         "Audio/ivector features/IVec_splitItem_fold_2_gmm_512_tvDim_20",
    #         "Audio/ivector features/IVec_splitItem_fold_2_gmm_512_tvDim_200",
    #         "Audio/ivector features/IVec_splitItem_fold_2_gmm_512_tvDim_40",
    #         "Audio/ivector features/IVec_splitItem_fold_2_gmm_512_tvDim_400",
    #         "Audio/ivector features/IVec_splitItem_fold_2_gmm_64_tvDim_10",
    #         "Audio/ivector features/IVec_splitItem_fold_2_gmm_64_tvDim_100",
    #         "Audio/ivector features/IVec_splitItem_fold_2_gmm_64_tvDim_20",
    #         "Audio/ivector features/IVec_splitItem_fold_2_gmm_64_tvDim_200",
    #         "Audio/ivector features/IVec_splitItem_fold_2_gmm_64_tvDim_40",
    #         "Audio/ivector features/IVec_splitItem_fold_2_gmm_64_tvDim_400",
    #         "Audio/ivector features/IVec_splitItem_fold_3_gmm_128_tvDim_10",
    #         "Audio/ivector features/IVec_splitItem_fold_3_gmm_128_tvDim_100",
    #         "Audio/ivector features/IVec_splitItem_fold_3_gmm_128_tvDim_20",
    #         "Audio/ivector features/IVec_splitItem_fold_3_gmm_128_tvDim_200",
    #         "Audio/ivector features/IVec_splitItem_fold_3_gmm_128_tvDim_40",
    #         "Audio/ivector features/IVec_splitItem_fold_3_gmm_128_tvDim_400",
    #         "Audio/ivector features/IVec_splitItem_fold_3_gmm_16_tvDim_10",
    #         "Audio/ivector features/IVec_splitItem_fold_3_gmm_16_tvDim_100",
    #         "Audio/ivector features/IVec_splitItem_fold_3_gmm_16_tvDim_20",
    #         "Audio/ivector features/IVec_splitItem_fold_3_gmm_16_tvDim_200",
    #         "Audio/ivector features/IVec_splitItem_fold_3_gmm_16_tvDim_40",
    #         "Audio/ivector features/IVec_splitItem_fold_3_gmm_16_tvDim_400",
    #         "Audio/ivector features/IVec_splitItem_fold_3_gmm_256_tvDim_10",
    #         "Audio/ivector features/IVec_splitItem_fold_3_gmm_256_tvDim_100",
    #         "Audio/ivector features/IVec_splitItem_fold_3_gmm_256_tvDim_20",
    #         "Audio/ivector features/IVec_splitItem_fold_3_gmm_256_tvDim_200",
    #         "Audio/ivector features/IVec_splitItem_fold_3_gmm_256_tvDim_40",
    #         "Audio/ivector features/IVec_splitItem_fold_3_gmm_256_tvDim_400",
    #         "Audio/ivector features/IVec_splitItem_fold_3_gmm_32_tvDim_10",
    #         "Audio/ivector features/IVec_splitItem_fold_3_gmm_32_tvDim_100",
    #         "Audio/ivector features/IVec_splitItem_fold_3_gmm_32_tvDim_20",
    #         "Audio/ivector features/IVec_splitItem_fold_3_gmm_32_tvDim_200",
    #         "Audio/ivector features/IVec_splitItem_fold_3_gmm_32_tvDim_40",
    #         "Audio/ivector features/IVec_splitItem_fold_3_gmm_32_tvDim_400",
    #         "Audio/ivector features/IVec_splitItem_fold_3_gmm_512_tvDim_10",
    #         "Audio/ivector features/IVec_splitItem_fold_3_gmm_512_tvDim_100",
    #         "Audio/ivector features/IVec_splitItem_fold_3_gmm_512_tvDim_20",
    #         "Audio/ivector features/IVec_splitItem_fold_3_gmm_512_tvDim_200",
    #         "Audio/ivector features/IVec_splitItem_fold_3_gmm_512_tvDim_40",
    #         "Audio/ivector features/IVec_splitItem_fold_3_gmm_512_tvDim_400",
    #         "Audio/ivector features/IVec_splitItem_fold_3_gmm_64_tvDim_10",
    #         "Audio/ivector features/IVec_splitItem_fold_3_gmm_64_tvDim_100",
    #         "Audio/ivector features/IVec_splitItem_fold_3_gmm_64_tvDim_20",
    #         "Audio/ivector features/IVec_splitItem_fold_3_gmm_64_tvDim_200",
    #         "Audio/ivector features/IVec_splitItem_fold_3_gmm_64_tvDim_40",
    #         "Audio/ivector features/IVec_splitItem_fold_3_gmm_64_tvDim_400",
    #         "Audio/ivector features/IVec_splitItem_fold_4_gmm_128_tvDim_10",
    #         "Audio/ivector features/IVec_splitItem_fold_4_gmm_128_tvDim_100",
    #         "Audio/ivector features/IVec_splitItem_fold_4_gmm_128_tvDim_20",
    #         "Audio/ivector features/IVec_splitItem_fold_4_gmm_128_tvDim_200",
    #         "Audio/ivector features/IVec_splitItem_fold_4_gmm_128_tvDim_40",
    #         "Audio/ivector features/IVec_splitItem_fold_4_gmm_128_tvDim_400",
    #         "Audio/ivector features/IVec_splitItem_fold_4_gmm_16_tvDim_10",
    #         "Audio/ivector features/IVec_splitItem_fold_4_gmm_16_tvDim_100",
    #         "Audio/ivector features/IVec_splitItem_fold_4_gmm_16_tvDim_20",
    #         "Audio/ivector features/IVec_splitItem_fold_4_gmm_16_tvDim_200",
    #         "Audio/ivector features/IVec_splitItem_fold_4_gmm_16_tvDim_40",
    #         "Audio/ivector features/IVec_splitItem_fold_4_gmm_16_tvDim_400",
    #         "Audio/ivector features/IVec_splitItem_fold_4_gmm_256_tvDim_10",
    #         "Audio/ivector features/IVec_splitItem_fold_4_gmm_256_tvDim_100",
    #         "Audio/ivector features/IVec_splitItem_fold_4_gmm_256_tvDim_20",
    #         "Audio/ivector features/IVec_splitItem_fold_4_gmm_256_tvDim_200",
    #         "Audio/ivector features/IVec_splitItem_fold_4_gmm_256_tvDim_40",
    #         "Audio/ivector features/IVec_splitItem_fold_4_gmm_256_tvDim_400",
    #         "Audio/ivector features/IVec_splitItem_fold_4_gmm_32_tvDim_10",
    #         "Audio/ivector features/IVec_splitItem_fold_4_gmm_32_tvDim_100",
    #         "Audio/ivector features/IVec_splitItem_fold_4_gmm_32_tvDim_20",
    #         "Audio/ivector features/IVec_splitItem_fold_4_gmm_32_tvDim_200",
    #         "Audio/ivector features/IVec_splitItem_fold_4_gmm_32_tvDim_40",
    #         "Audio/ivector features/IVec_splitItem_fold_4_gmm_32_tvDim_400",
    #         "Audio/ivector features/IVec_splitItem_fold_4_gmm_512_tvDim_10",
    #         "Audio/ivector features/IVec_splitItem_fold_4_gmm_512_tvDim_100",
    #         "Audio/ivector features/IVec_splitItem_fold_4_gmm_512_tvDim_20",
    #         "Audio/ivector features/IVec_splitItem_fold_4_gmm_512_tvDim_200",
    #         "Audio/ivector features/IVec_splitItem_fold_4_gmm_512_tvDim_40",
    #         "Audio/ivector features/IVec_splitItem_fold_4_gmm_512_tvDim_400",
    #         "Audio/ivector features/IVec_splitItem_fold_4_gmm_64_tvDim_10",
    #         "Audio/ivector features/IVec_splitItem_fold_4_gmm_64_tvDim_100",
    #         "Audio/ivector features/IVec_splitItem_fold_4_gmm_64_tvDim_20",
    #         "Audio/ivector features/IVec_splitItem_fold_4_gmm_64_tvDim_200",
    #         "Audio/ivector features/IVec_splitItem_fold_4_gmm_64_tvDim_40",
    #         "Audio/ivector features/IVec_splitItem_fold_4_gmm_64_tvDim_400",
    #         "Audio/ivector features/IVec_splitItem_fold_5_gmm_128_tvDim_10",
    #         "Audio/ivector features/IVec_splitItem_fold_5_gmm_128_tvDim_100",
    #         "Audio/ivector features/IVec_splitItem_fold_5_gmm_128_tvDim_20",
    #         "Audio/ivector features/IVec_splitItem_fold_5_gmm_128_tvDim_200",
    #         "Audio/ivector features/IVec_splitItem_fold_5_gmm_128_tvDim_40",
    #         "Audio/ivector features/IVec_splitItem_fold_5_gmm_128_tvDim_400",
    #         "Audio/ivector features/IVec_splitItem_fold_5_gmm_16_tvDim_10",
    #         "Audio/ivector features/IVec_splitItem_fold_5_gmm_16_tvDim_100",
    #         "Audio/ivector features/IVec_splitItem_fold_5_gmm_16_tvDim_20",
    #         "Audio/ivector features/IVec_splitItem_fold_5_gmm_16_tvDim_200",
    #         "Audio/ivector features/IVec_splitItem_fold_5_gmm_16_tvDim_40",
    #         "Audio/ivector features/IVec_splitItem_fold_5_gmm_16_tvDim_400",
    #         "Audio/ivector features/IVec_splitItem_fold_5_gmm_256_tvDim_10",
    #         "Audio/ivector features/IVec_splitItem_fold_5_gmm_256_tvDim_100",
    #         "Audio/ivector features/IVec_splitItem_fold_5_gmm_256_tvDim_20",
    #         "Audio/ivector features/IVec_splitItem_fold_5_gmm_256_tvDim_200",
    #         "Audio/ivector features/IVec_splitItem_fold_5_gmm_256_tvDim_40",
    #         "Audio/ivector features/IVec_splitItem_fold_5_gmm_256_tvDim_400",
    #         "Audio/ivector features/IVec_splitItem_fold_5_gmm_32_tvDim_10",
    #         "Audio/ivector features/IVec_splitItem_fold_5_gmm_32_tvDim_100",
    #         "Audio/ivector features/IVec_splitItem_fold_5_gmm_32_tvDim_20",
    #         "Audio/ivector features/IVec_splitItem_fold_5_gmm_32_tvDim_200",
    #         "Audio/ivector features/IVec_splitItem_fold_5_gmm_32_tvDim_40",
    #         "Audio/ivector features/IVec_splitItem_fold_5_gmm_32_tvDim_400",
    #         "Audio/ivector features/IVec_splitItem_fold_5_gmm_512_tvDim_10",
    #         "Audio/ivector features/IVec_splitItem_fold_5_gmm_512_tvDim_100",
    #         "Audio/ivector features/IVec_splitItem_fold_5_gmm_512_tvDim_20",
    #         "Audio/ivector features/IVec_splitItem_fold_5_gmm_512_tvDim_200",
    #         "Audio/ivector features/IVec_splitItem_fold_5_gmm_512_tvDim_40",
    #         "Audio/ivector features/IVec_splitItem_fold_5_gmm_512_tvDim_400",
    #         "Audio/ivector features/IVec_splitItem_fold_5_gmm_64_tvDim_10",
    #         "Audio/ivector features/IVec_splitItem_fold_5_gmm_64_tvDim_100",
    #         "Audio/ivector features/IVec_splitItem_fold_5_gmm_64_tvDim_20",
    #         "Audio/ivector features/IVec_splitItem_fold_5_gmm_64_tvDim_200",
    #         "Audio/ivector features/IVec_splitItem_fold_5_gmm_64_tvDim_40",
    #         "Audio/ivector features/IVec_splitItem_fold_5_gmm_64_tvDim_400",
    #         "Data/itemids_splitted_5foldCV/useritemIds_test_itemsplit_fold1of5",
    #         "Data/itemids_splitted_5foldCV/useritemIds_test_itemsplit_fold2of5",
    #         "Data/itemids_splitted_5foldCV/useritemIds_test_itemsplit_fold3of5",
    #         "Data/itemids_splitted_5foldCV/useritemIds_test_itemsplit_fold4of5",
    #         "Data/itemids_splitted_5foldCV/useritemIds_test_itemsplit_fold5of5",
    #         "Data/itemids_splitted_5foldCV/useritemIds_train_itemsplit_fold1of5",
    #         "Data/itemids_splitted_5foldCV/useritemIds_train_itemsplit_fold2of5",
    #         "Data/itemids_splitted_5foldCV/useritemIds_train_itemsplit_fold3of5",
    #         "Data/itemids_splitted_5foldCV/useritemIds_train_itemsplit_fold4of5",
    #         "Data/itemids_splitted_5foldCV/useritemIds_train_itemsplit_fold5of5",
    #         "Data/movie_description",
    #         "Metadata/GenreFeatures",
    #         "Metadata/TagFeatures",
    #         "Metadata/YearOfProd",
    #         "Visual/Aesthetic features/AvgVar/AestheticFeatures - AVGVAR - All",
    #         "Visual/Aesthetic features/AvgVar/AestheticFeatures - AVGVAR - Feat26avgHSLFocus",
    #         "Visual/Aesthetic features/AvgVar/AestheticFeatures - AVGVAR - Feat26avgHSVRot",
    #         "Visual/Aesthetic features/AvgVar/AestheticFeatures - AVGVAR - Feat26avgWaveletHSV",
    #         "Visual/Aesthetic features/AvgVar/AestheticFeatures - AVGVAR - Feat26Brightness",
    #         "Visual/Aesthetic features/AvgVar/AestheticFeatures - AVGVAR - Feat26BrightnessSegments",
    #         "Visual/Aesthetic features/AvgVar/AestheticFeatures - AVGVAR - Feat26Centroids",
    #         "Visual/Aesthetic features/AvgVar/AestheticFeatures - AVGVAR - Feat26Colorfulness",
    #         "Visual/Aesthetic features/AvgVar/AestheticFeatures - AVGVAR - Feat26ColorModel",
    #         "Visual/Aesthetic features/AvgVar/AestheticFeatures - AVGVAR - Feat26ContrastSegments",
    #         "Visual/Aesthetic features/AvgVar/AestheticFeatures - AVGVAR - Feat26Convexity",
    #         "Visual/Aesthetic features/AvgVar/AestheticFeatures - AVGVAR - Feat26Coordinates",
    #         "Visual/Aesthetic features/AvgVar/AestheticFeatures - AVGVAR - Feat26DofIndicator",
    #         "Visual/Aesthetic features/AvgVar/AestheticFeatures - AVGVAR - Feat26Edge",
    #         "Visual/Aesthetic features/AvgVar/AestheticFeatures - AVGVAR - Feat26HSL",
    #         "Visual/Aesthetic features/AvgVar/AestheticFeatures - AVGVAR - Feat26HSV",
    #         "Visual/Aesthetic features/AvgVar/AestheticFeatures - AVGVAR - Feat26HueDescriptors",
    #         "Visual/Aesthetic features/AvgVar/AestheticFeatures - AVGVAR - Feat26HueModels",
    #         "Visual/Aesthetic features/AvgVar/AestheticFeatures - AVGVAR - Feat26HueSegments",
    #         "Visual/Aesthetic features/AvgVar/AestheticFeatures - AVGVAR - Feat26LargeSegments",
    #         "Visual/Aesthetic features/AvgVar/AestheticFeatures - AVGVAR - Feat26MassVarianceSegments",
    #         "Visual/Aesthetic features/AvgVar/AestheticFeatures - AVGVAR - Feat26RGBEntropy",
    #         "Visual/Aesthetic features/AvgVar/AestheticFeatures - AVGVAR - Feat26SaturationSegments",
    #         "Visual/Aesthetic features/AvgVar/AestheticFeatures - AVGVAR - Feat26SkewnessSegments",
    #         "Visual/Aesthetic features/AvgVar/AestheticFeatures - AVGVAR - Feat26Texture",
    #         "Visual/Aesthetic features/AvgVar/AestheticFeatures - AVGVAR - Feat26ValueSegments",
    #         "Visual/Aesthetic features/AvgVar/AestheticFeatures - AVGVAR - Feat26WaveletHSV",
    #         "Visual/Aesthetic features/AvgVar/AestheticFeatures - AVGVAR - Type3Color",
    #         "Visual/Aesthetic features/AvgVar/AestheticFeatures - AVGVAR - Type3Object",
    #         "Visual/Aesthetic features/AvgVar/AestheticFeatures - AVGVAR - Type3Texture",
    #         "Visual/Aesthetic features/Avg/AestheticFeatures - AVG - All",
    #         "Visual/Aesthetic features/Avg/AestheticFeatures - AVG - Feat26avgHSLFocus",
    #         "Visual/Aesthetic features/Avg/AestheticFeatures - AVG - Feat26avgHSVRot",
    #         "Visual/Aesthetic features/Avg/AestheticFeatures - AVG - Feat26avgWaveletHSV",
    #         "Visual/Aesthetic features/Avg/AestheticFeatures - AVG - Feat26Brightness",
    #         "Visual/Aesthetic features/Avg/AestheticFeatures - AVG - Feat26BrightnessSegments",
    #         "Visual/Aesthetic features/Avg/AestheticFeatures - AVG - Feat26Centroids",
    #         "Visual/Aesthetic features/Avg/AestheticFeatures - AVG - Feat26Colorfulness",
    #         "Visual/Aesthetic features/Avg/AestheticFeatures - AVG - Feat26ColorModel",
    #         "Visual/Aesthetic features/Avg/AestheticFeatures - AVG - Feat26ContrastSegments",
    #         "Visual/Aesthetic features/Avg/AestheticFeatures - AVG - Feat26Convexity",
    #         "Visual/Aesthetic features/Avg/AestheticFeatures - AVG - Feat26Coordinates",
    #         "Visual/Aesthetic features/Avg/AestheticFeatures - AVG - Feat26DofIndicator",
    #         "Visual/Aesthetic features/Avg/AestheticFeatures - AVG - Feat26Edge",
    #         "Visual/Aesthetic features/Avg/AestheticFeatures - AVG - Feat26HSL",
    #         "Visual/Aesthetic features/Avg/AestheticFeatures - AVG - Feat26HSV",
    #         "Visual/Aesthetic features/Avg/AestheticFeatures - AVG - Feat26HueDescriptors",
    #         "Visual/Aesthetic features/Avg/AestheticFeatures - AVG - Feat26HueModels",
    #         "Visual/Aesthetic features/Avg/AestheticFeatures - AVG - Feat26HueSegments",
    #         "Visual/Aesthetic features/Avg/AestheticFeatures - AVG - Feat26LargeSegments",
    #         "Visual/Aesthetic features/Avg/AestheticFeatures - AVG - Feat26MassVarianceSegments",
    #         "Visual/Aesthetic features/Avg/AestheticFeatures - AVG - Feat26RGBEntropy",
    #         "Visual/Aesthetic features/Avg/AestheticFeatures - AVG - Feat26SaturationSegments",
    #         "Visual/Aesthetic features/Avg/AestheticFeatures - AVG - Feat26SkewnessSegments",
    #         "Visual/Aesthetic features/Avg/AestheticFeatures - AVG - Feat26Texture",
    #         "Visual/Aesthetic features/Avg/AestheticFeatures - AVG - Feat26ValueSegments",
    #         "Visual/Aesthetic features/Avg/AestheticFeatures - AVG - Feat26WaveletHSV",
    #         "Visual/Aesthetic features/Avg/AestheticFeatures - AVG - Type3Color",
    #         "Visual/Aesthetic features/Avg/AestheticFeatures - AVG - Type3Object",
    #         "Visual/Aesthetic features/Avg/AestheticFeatures - AVG - Type3Texture",
    #         "Visual/Aesthetic features/MedMad/AestheticFeatures - MEDMAD - All",
    #         "Visual/Aesthetic features/MedMad/AestheticFeatures - MEDMAD - Feat26avgHSLFocus",
    #         "Visual/Aesthetic features/MedMad/AestheticFeatures - MEDMAD - Feat26avgHSVRot",
    #         "Visual/Aesthetic features/MedMad/AestheticFeatures - MEDMAD - Feat26avgWaveletHSV",
    #         "Visual/Aesthetic features/MedMad/AestheticFeatures - MEDMAD - Feat26Brightness",
    #         "Visual/Aesthetic features/MedMad/AestheticFeatures - MEDMAD - Feat26BrightnessSegments",
    #         "Visual/Aesthetic features/MedMad/AestheticFeatures - MEDMAD - Feat26Centroids",
    #         "Visual/Aesthetic features/MedMad/AestheticFeatures - MEDMAD - Feat26Colorfulness",
    #         "Visual/Aesthetic features/MedMad/AestheticFeatures - MEDMAD - Feat26ColorModel",
    #         "Visual/Aesthetic features/MedMad/AestheticFeatures - MEDMAD - Feat26ContrastSegments",
    #         "Visual/Aesthetic features/MedMad/AestheticFeatures - MEDMAD - Feat26Convexity",
    #         "Visual/Aesthetic features/MedMad/AestheticFeatures - MEDMAD - Feat26Coordinates",
    #         "Visual/Aesthetic features/MedMad/AestheticFeatures - MEDMAD - Feat26DofIndicator",
    #         "Visual/Aesthetic features/MedMad/AestheticFeatures - MEDMAD - Feat26Edge",
    #         "Visual/Aesthetic features/MedMad/AestheticFeatures - MEDMAD - Feat26HSL",
    #         "Visual/Aesthetic features/MedMad/AestheticFeatures - MEDMAD - Feat26HSV",
    #         "Visual/Aesthetic features/MedMad/AestheticFeatures - MEDMAD - Feat26HueDescriptors",
    #         "Visual/Aesthetic features/MedMad/AestheticFeatures - MEDMAD - Feat26HueModels",
    #         "Visual/Aesthetic features/MedMad/AestheticFeatures - MEDMAD - Feat26HueSegments",
    #         "Visual/Aesthetic features/MedMad/AestheticFeatures - MEDMAD - Feat26LargeSegments",
    #         "Visual/Aesthetic features/MedMad/AestheticFeatures - MEDMAD - Feat26MassVarianceSegments",
    #         "Visual/Aesthetic features/MedMad/AestheticFeatures - MEDMAD - Feat26RGBEntropy",
    #         "Visual/Aesthetic features/MedMad/AestheticFeatures - MEDMAD - Feat26SaturationSegments",
    #         "Visual/Aesthetic features/MedMad/AestheticFeatures - MEDMAD - Feat26SkewnessSegments",
    #         "Visual/Aesthetic features/MedMad/AestheticFeatures - MEDMAD - Feat26Texture",
    #         "Visual/Aesthetic features/MedMad/AestheticFeatures - MEDMAD - Feat26ValueSegments",
    #         "Visual/Aesthetic features/MedMad/AestheticFeatures - MEDMAD - Feat26WaveletHSV",
    #         "Visual/Aesthetic features/MedMad/AestheticFeatures - MEDMAD - Type3Color",
    #         "Visual/Aesthetic features/MedMad/AestheticFeatures - MEDMAD - Type3Object",
    #         "Visual/Aesthetic features/MedMad/AestheticFeatures - MEDMAD - Type3Texture",
    #         "Visual/Aesthetic features/Med/AestheticFeatures - MED - All",
    #         "Visual/Aesthetic features/Med/AestheticFeatures - MED - Feat26avgHSLFocus",
    #         "Visual/Aesthetic features/Med/AestheticFeatures - MED - Feat26avgHSVRot",
    #         "Visual/Aesthetic features/Med/AestheticFeatures - MED - Feat26avgWaveletHSV",
    #         "Visual/Aesthetic features/Med/AestheticFeatures - MED - Feat26Brightness",
    #         "Visual/Aesthetic features/Med/AestheticFeatures - MED - Feat26BrightnessSegments",
    #         "Visual/Aesthetic features/Med/AestheticFeatures - MED - Feat26Centroids",
    #         "Visual/Aesthetic features/Med/AestheticFeatures - MED - Feat26Colorfulness",
    #         "Visual/Aesthetic features/Med/AestheticFeatures - MED - Feat26ColorModel",
    #         "Visual/Aesthetic features/Med/AestheticFeatures - MED - Feat26ContrastSegments",
    #         "Visual/Aesthetic features/Med/AestheticFeatures - MED - Feat26Convexity",
    #         "Visual/Aesthetic features/Med/AestheticFeatures - MED - Feat26Coordinates",
    #         "Visual/Aesthetic features/Med/AestheticFeatures - MED - Feat26DofIndicator",
    #         "Visual/Aesthetic features/Med/AestheticFeatures - MED - Feat26Edge",
    #         "Visual/Aesthetic features/Med/AestheticFeatures - MED - Feat26HSL",
    #         "Visual/Aesthetic features/Med/AestheticFeatures - MED - Feat26HSV",
    #         "Visual/Aesthetic features/Med/AestheticFeatures - MED - Feat26HueDescriptors",
    #         "Visual/Aesthetic features/Med/AestheticFeatures - MED - Feat26HueModels",
    #         "Visual/Aesthetic features/Med/AestheticFeatures - MED - Feat26HueSegments",
    #         "Visual/Aesthetic features/Med/AestheticFeatures - MED - Feat26LargeSegments",
    #         "Visual/Aesthetic features/Med/AestheticFeatures - MED - Feat26MassVarianceSegments",
    #         "Visual/Aesthetic features/Med/AestheticFeatures - MED - Feat26RGBEntropy",
    #         "Visual/Aesthetic features/Med/AestheticFeatures - MED - Feat26SaturationSegments",
    #         "Visual/Aesthetic features/Med/AestheticFeatures - MED - Feat26SkewnessSegments",
    #         "Visual/Aesthetic features/Med/AestheticFeatures - MED - Feat26Texture",
    #         "Visual/Aesthetic features/Med/AestheticFeatures - MED - Feat26ValueSegments",
    #         "Visual/Aesthetic features/Med/AestheticFeatures - MED - Feat26WaveletHSV",
    #         "Visual/Aesthetic features/Med/AestheticFeatures - MED - Type3Color",
    #         "Visual/Aesthetic features/Med/AestheticFeatures - MED - Type3Object",
    #         "Visual/Aesthetic features/Med/AestheticFeatures - MED - Type3Texture",
    #         "Visual/AlexNet features/AvgVar/AlexNetFeatures - AVGVAR - fc7",
    #         "Visual/AlexNet features/Avg/AlexNetFeatures - AVG - fc7",
    #         "Visual/AlexNet features/MedMad/AlexNetFeatures - MEDMAD - fc7",
    #         "Visual/AlexNet features/Med/AlexNetFeatures - MED - fc7",
    #     ]



    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        self.zip_file_folder = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER
        self.decompressed_zip_file_folder = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            self.dataFile = zipfile.ZipFile(self.zip_file_folder + "Final_MMTF14K_Web.zip","r")

        except (FileNotFoundError, zipfile.BadZipFile):

            self._print("Unable to find data zip file.")
            self._print("Automatic download not available, please ensure the ZIP data file is in folder {}.".format(self.zip_file_folder))
            self._print("Data can be downloaded here: {}".format(self.DATASET_URL))

            # If directory does not exist, create
            if not os.path.exists(self.zip_file_folder):
                os.makedirs(self.zip_file_folder)

            raise FileNotFoundError("Automatic download not available.")




        # Load item list to create mapper
        self.item_original_ID_to_index = self._load_available_items()


        movielens20M = Movielens20MReader()
        movielens20M = movielens20M.load_data()

        URM_all = movielens20M.get_URM_all()
        n_users, n_items = URM_all.shape

        # Remove items with no features
        movielens_items_to_keep = []

        for movielens_item_id, movielens_item_index in movielens20M.get_item_original_ID_to_index_mapper().items():

            if movielens_item_id in self.item_original_ID_to_index:
                movielens_items_to_keep.append(movielens_item_index)

        URM_all = URM_all[:,movielens_items_to_keep]

        # Remove users with no interactions
        URM_all = sps.csr_matrix(URM_all)
        users_to_keep = np.ediff1d(URM_all.indptr)>0

        URM_all = URM_all[users_to_keep,:]

        self.user_original_ID_to_index = movielens20M.get_user_original_ID_to_index_mapper().copy()

        self.user_original_ID_to_index = reconcile_mapper_with_removed_tokens(self.user_original_ID_to_index,
                                                                              np.arange(n_users, dtype=np.int)[np.logical_not(users_to_keep)])






        ICM_name_to_path = {}
        ICM_name_to_path["ICM_Tag"] = "Metadata/TagFeatures.csv"
        ICM_name_to_path["ICM_Genre"] = "Metadata/GenreFeatures.csv"
        ICM_name_to_path["ICM_Year"] = "Metadata/YearOfProd.csv"


        # Load BLF features
        BLF_file_name_list = [
            "BLF_Correlation",
            "BLF_DeltaSpectral",
            "BLF_LogarithmicFluctuation",
            "BLF_SpectralContrast",
            "BLF_Spectral",
            "BLF_VarianceDeltaSpectral"
        ]

        for ICM_name in BLF_file_name_list:

            file_path = "Audio/Block level features/Component6/" + ICM_name.upper() + "feat.csv"
            ICM_name_to_path["ICM_" + ICM_name] = file_path


        for gmm in [16, 32, 64, 128, 256, 512]:
            for tv_dim in [10, 20, 40, 100, 200, 400]:

                file_path = "Audio/ivector features/IVec_splitItem_fold_1_gmm_{}_tvDim_{}.csv".format(gmm, tv_dim)

                ICM_name = "ICM_Ivector_gmm_{}_tvDim_{}".format(gmm, tv_dim)
                ICM_name_to_path[ICM_name] = file_path


        for aggregation in ["Avg", "AvgVar", "Med", "MedMad"]:

            file_path = "Visual/AlexNet features/{}/AlexNetFeatures - {} - fc7.csv".format(aggregation, aggregation.upper())

            ICM_name = "ICM_AlexNet_{}".format(aggregation)
            ICM_name_to_path[ICM_name] = file_path

            for aesthetic_type in ['All']:
                                   #  , 'Feat26avgHSLFocus', 'Feat26avgHSVRot', 'Feat26avgWaveletHSV', 'Feat26Brightness',
                                   # 'Feat26BrightnessSegments', 'Feat26Centroids', 'Feat26Colorfulness', 'Feat26ColorModel',
                                   # 'Feat26ContrastSegments', 'Feat26Convexity', 'Feat26Coordinates', 'Feat26DofIndicator',
                                   # 'Feat26Edge', 'Feat26HSL', 'Feat26HSV', 'Feat26HueDescriptors', 'Feat26HueModels',
                                   # 'Feat26HueSegments', 'Feat26LargeSegments', 'Feat26MassVarianceSegments', 'Feat26RGBEntropy',
                                   # 'Feat26SaturationSegments', 'Feat26SkewnessSegments', 'Feat26Texture', 'Feat26ValueSegments',
                                   # 'Feat26WaveletHSV', 'Type3Color', 'Type3Object', 'Type3Texture']:

                file_path = "Visual/Aesthetic features/{}/AestheticFeatures - {} - {}.csv".format(aggregation, aggregation.upper(), aesthetic_type)

                ICM_name = "ICM_Aesthetic_{}_{}".format(aggregation, aesthetic_type)
                ICM_name_to_path[ICM_name] = file_path


        loaded_ICM_dict = {}
        loaded_ICM_mapper_dict = {}

        for ICM_count, (ICM_name, ICM_path) in enumerate(ICM_name_to_path.items()):

            self._print("Loading Item Features {} of {}: '{}'".format(ICM_count+1, len(ICM_name_to_path), ICM_name))

            ICM_path = self.dataFile.extract(ICM_path, path=self.decompressed_zip_file_folder + "decompressed/")

            ICM_object, ICM_mapper_object, _ = self._loadICM_CSV(self.item_original_ID_to_index, ICM_path, header=True, separator=',')
            ICM_object = sps.csr_matrix(ICM_object)

            loaded_ICM_dict[ICM_name] = ICM_object
            loaded_ICM_mapper_dict[ICM_name] = ICM_mapper_object

            shutil.rmtree(self.decompressed_zip_file_folder + "decompressed/", ignore_errors=True)


        loaded_URM_dict = {"URM_all": URM_all}

        loaded_dataset = Dataset(dataset_name = self._get_dataset_name(),
                                 URM_dictionary = loaded_URM_dict,
                                 ICM_dictionary = loaded_ICM_dict,
                                 ICM_feature_mapper_dictionary = loaded_ICM_mapper_dict,
                                 UCM_dictionary = None,
                                 UCM_feature_mapper_dictionary = None,
                                 user_original_ID_to_index= self.user_original_ID_to_index,
                                 item_original_ID_to_index= self.item_original_ID_to_index,
                                 is_implicit = self.IS_IMPLICIT,
                                 )

        self._print("Loading Complete")

        return loaded_dataset





    def _load_available_items(self, header=True, separator=','):

        item_list_path = "Data/movie_description.csv"
        item_list_path = self.dataFile.extract(item_list_path, path=self.decompressed_zip_file_folder + "decompressed/")

        item_original_ID_to_index = {}

        readCSV = csv.reader(open(item_list_path, "r", encoding="latin1"), delimiter=separator)

        if header:
            readCSV.__next__()

        for line in readCSV:

            item_id = line[0]

            if item_id not in item_original_ID_to_index:
                item_original_ID_to_index[item_id] = len(item_original_ID_to_index)


        return item_original_ID_to_index





    def _loadICM_CSV(self, item_original_ID_to_index, file_path, header=True, separator=','):

        readCSV = csv.reader(open(file_path, "r", encoding="latin1"), delimiter=separator)

        ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_row_mapper=item_original_ID_to_index.copy(),
                                                        on_new_row="ignore",
                                                        preinitialized_col_mapper={},
                                                        on_new_col="add",
                                                        dtype=np.float32)

        numCells = 0
        non_acceptable_values_count = 0

        feature_name = None

        if header:
            feature_name = readCSV.__next__()
            feature_name = feature_name[1:]
            feature_name = np.array(feature_name)


        for line in readCSV:

            numCells += 1

            if (numCells % 1000000 == 0):
                print("Processed {} rows".format(numCells))

            movie_id = line[0]

            item_features = line[1:]
            item_features = np.array(item_features, dtype=float)

            if feature_name is None:
                feature_name = np.arange(0, len(item_features), dtype=int)

            non_acceptable_values_mask = np.logical_or(np.isinf(item_features), np.isnan(item_features))
            item_features[non_acceptable_values_mask] = 0.0

            non_acceptable_values_count += non_acceptable_values_mask.sum()
            nonzero_mask = item_features != 0.0

            ICM_builder.add_data_lists([movie_id]*nonzero_mask.sum(), feature_name[nonzero_mask], item_features[nonzero_mask])


        if non_acceptable_values_count != 0:
            print("MultifacetedMovieTrailerFeatureReader: Non numeric values in ICM found, inf and nan: {}. Setting them as 0.0".format(non_acceptable_values_count))

        return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()
