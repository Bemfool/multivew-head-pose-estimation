#ifndef HPE_CONSTANT_H
#define HPE_CONSTANT_H

#include <assert.h>
#include <string>

enum Status {
	Status_Error = 0L,
	Status_Ok = 1L,
};

enum RotateType
{
	RotateType_Invalid = 0L, 
	RotateType_CW, 
	RotateType_CCW,
};

const std::string LOG_PATH = R"(./log)";

const unsigned int N_PHOTOS = 24u;
const unsigned int N_VERTICES = 46990u;
const unsigned int N_FACES = 93322u;
const unsigned int N_ID_PCS = 99u;
const unsigned int N_EXPR_PCS = 29u;
const unsigned int N_LANDMARKS = 68u;
const unsigned int N_RES = N_LANDMARKS * 2 * N_PHOTOS;

const unsigned int START = 4u;
const unsigned int END = 24u;

const double ZOOM_SCALE =  0.1;

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
	SolveExtParamsMode_Default = 1L << 4,
};


#endif