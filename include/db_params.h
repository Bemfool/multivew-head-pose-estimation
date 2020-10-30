#pragma once

#include <string>

const std::string PROJECT_PATH = "/home/bemfoo/Data/face_zzm/project";
const std::string BFM_H5_PATH = "/home/bemfoo/Data/BaselFaceModel_mod.h5";
const std::string LANDMARK_IDX_PATH = "/home/bemfoo/Project/head-pose-estimation/data/example_landmark_68.anl";
const std::string DLIB_LANDMARK_DETECTOR_DATA_PATH = "/home/bemfoo/Data/shape_predictor_68_face_landmarks.dat";

const std::string SHAPE_MU_H5_PATH = "shapeMU";
const std::string SHAPE_EV_H5_PATH = "shapeEV";
const std::string SHAPE_PC_H5_PATH = "shapePC";
const std::string TEX_MU_H5_PATH = "texMU";
const std::string TEX_EV_H5_PATH = "texEV";
const std::string TEX_PC_H5_PATH = "texPC";
const std::string EXPR_MU_H5_PATH = "expMU";
const std::string EXPR_EV_H5_PATH = "expEV";
const std::string EXPR_PC_H5_PATH = "expPC";
const std::string TRIANGLE_LIST_H5_PATH = "faces";

const unsigned int N_PHOTOS = 24;
const unsigned int N_VERTICES = 46990;
const unsigned int N_FACES = 93322;
const unsigned int N_ID_PCS = 99;
const unsigned int N_EXPR_PCS = 29;
const unsigned int N_LANDMARKS = 68;

const double ZOOM_SCALE = 0.1;

const int N_CERES_ITERATIONS = 500;
const int N_CERES_THREADS = 16;
#ifndef _DEBUG
const bool B_CERES_STDCOUT = true;
#else
const bool B_CERES_STDCOUT = false;
#endif

enum SolveExtParamsMode
{
	SolveExtParamsMode_InvalidFirst = 0L << 0,
	SolveExtParamsMode_UseCeres = 1L << 0,
	SolveExtParamsMode_UseOpenCV = 1L << 1,
	SolveExtParamsMode_UseLinearizedRadians = 1L << 2,
	SolveExtParamsMode_UseDlt = 1L << 3,
};
