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
#include <memory>

#include "ceres/ceres.h"

#include "tiny_progress.hpp"

#include "glog/logging.h"

#include "string_utils.h"
#include "io_utils.h"
#include "functor/functor.h"
#include "texture.h"
#include "data_manager.h"


using Eigen::Matrix;
using Eigen::Vector3d;
using Eigen::VectorXd;
using Summary = ceres::Solver::Summary;

#define CERES_INIT(options, N_ITERATIONS, N_THREADS, B_STDCOUT) \
		options.max_num_iterations = N_ITERATIONS; \
 		options.num_threads = N_THREADS; \
		options.minimizer_progress_to_stdout = B_STDCOUT; 


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
		const std::string& sProjectPath,
        const std::string& sBfmH5Path,
        const std::string& sLandmarkIdxPath,
		const std::string& sDlibDetPath
	);
	Status init(
		const std::string& sProjectPath,
        const std::string& sBfmH5Path,
        const std::string& sLandmarkIdxPath,
		const std::string& sDlibDetPath
	);

	// inline BfmManager *getModel() { return m_pBfmManager; }
	inline std::shared_ptr<BfmManager>& getBfmManager() { return m_pBfmManager; }

	void solve(SolveExtParamsMode mode);
	
	Summary estExtParams(const DetPairVector&  aObjDets);
	// Summary estShapeCoef(const DetPairVector&  aObjDets);
	// Summary estExprCoef(const DetPairVector&  aObjDets);
	
	bool solveExtParams(long mode = SolveExtParamsMode_UseCeres, double ca = 1.0, double cb = 0.0);

	bool solveShapeCoef();


	bool solveExprCoef();


private:

	bool is_close_enough(double *ext_params, double rotation_eps = 0, double translation_eps = 0);

	std::vector<double> estInit3dPts(const std::vector<DetPair>& vDetPairs);
	void estInitSc(const std::vector<double>& vPts);
	void estInitExtParams(std::vector<double>& vPts);
	void initWin(
		dlib::image_window& window, 
		const std::string& sTitle, 
		const Arr2d& img,
		const ObjDet& objDet,
		bool bIsValid
	);
	void rmOutliers();

	shared_ptr<BfmManager> m_pBfmManager;
	shared_ptr<DataManager> m_pDataManager;

	std::vector<unsigned int> m_aLandmarkMap;
};


#endif // HPE_PROBLEM_H