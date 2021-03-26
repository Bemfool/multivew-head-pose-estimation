#ifndef HPE_PROBLEM_H
#define HPE_PROBLEM_H

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>

#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <fstream>
#include <assert.h>

#include "ceres/ceres.h"

#include "glog/logging.h"

#include "string_utils.h"
#include "io_utils.h"
#include "functor/functor.h"
#include "texture.h"
#include "data_manager.h"


using FullObjectDetection = dlib::full_object_detection;
using DetPairVector = std::vector<std::pair<FullObjectDetection, int>>;
using Eigen::Matrix;
using Eigen::Vector3d;
using Eigen::VectorXd;
using Summary = ceres::Solver::Summary;


#define CERES_INIT(N_ITERATIONS, N_THREADS, B_STDCOUT) \
		ceres::Solver::Options options; \
		options.max_num_iterations = N_ITERATIONS; \
 		options.num_threads = N_THREADS; \
		options.minimizer_progress_to_stdout = B_STDCOUT; \
		ceres::Solver::Summary summary;


class MHPEProblem
{


public:


	/* Initialization 
	 * Usage:
	 *     HeadPoseEstimationProblem hpe_problem;
	 *     HeadPoseEstimationProblem hpe_problem(filename);
	 * Parameters:
	 * 	   @filename: Filename for Basel Face Model loader.
	 *******************************************************************************
	 * Init a head pose estimation problem with input filename as bfm input filename.
	 */

	MHPEProblem(
		const std::string& strProjectPath,
        const std::string& strBfmH5Path,
        const std::string& strLandmarkIdxPath
	);
	Status init(
		const std::string& strProjectPath,
        const std::string& strBfmH5Path,
        const std::string& strLandmarkIdxPath
	);

	inline BaselFaceModelManager *getModel() { return m_pModel; }

	void solve(SolveExtParamsMode mode, const std::string& strDlibDetPath);
	
	void estExtParams(double *pPoints, double dScMean);
	Summary estExtParams(const DetPairVector&  aObjDets);
	Summary estShapeCoef(const DetPairVector&  aObjDets);
	Summary estExprCoef(const DetPairVector&  aObjDets);
	
	inline void setObservedPoints(FullObjectDetection *pObservedPoints) { m_pObservedPoints = pObservedPoints; }

	bool solveExtParams(long mode = SolveExtParamsMode_UseCeres, double ca = 1.0, double cb = 0.0);

	bool solveShapeCoef();


	bool solveExprCoef();


private:

	bool is_close_enough(double *ext_params, double rotation_eps = 0, double translation_eps = 0);
	
	std::vector<std::pair<FullObjectDetection, int>> m_aObjDetections;
	FullObjectDetection* m_pObservedPoints;
	BaselFaceModelManager *m_pModel;
	DataManager* m_pDataManager;
	std::vector<unsigned int> m_aLandmarkMap;

};


#endif // HPE_PROBLEM_H