#include "hpe_problem.h"

HeadPoseEstimationProblem::HeadPoseEstimationProblem() {
	init();
}

BfmStatus HeadPoseEstimationProblem::init() {
	BFM_DEBUG(PRINT_GREEN "#################### Init Head Pose Esitimation Problem ####################\n" COLOR_END);

	m_pDataManager = new DataManager(PROJECT_PATH);

	double aIntParams[4];
	aIntParams[0] = aIntParams[1] = m_pDataManager->getF() * ZOOM_SCALE;
	aIntParams[2] = (m_pDataManager->getWidth() * 0.5 + m_pDataManager->getCx()) * ZOOM_SCALE;
	aIntParams[3] = (m_pDataManager->getHeight() * 0.5 + m_pDataManager->getCy()) * ZOOM_SCALE;
	std::cout << "Intrinsic Parameters: " << aIntParams[0] << " " << aIntParams[2] << " " << aIntParams[3] << std::endl;

	m_pModel = new BaselFaceModelManager(
		BFM_H5_PATH,
		N_VERTICES, N_FACES, N_ID_PCS, N_EXPR_PCS,
		aIntParams,
		SHAPE_MU_H5_PATH, SHAPE_EV_H5_PATH, SHAPE_PC_H5_PATH,
		TEX_MU_H5_PATH, TEX_EV_H5_PATH, TEX_PC_H5_PATH,
		EXPR_MU_H5_PATH, EXPR_EV_H5_PATH, EXPR_PC_H5_PATH,
		TRIANGLE_LIST_H5_PATH,
	N_LANDMARKS,
		LANDMARK_IDX_PATH
	);

	std::fill(m_aExtParams, m_aExtParams + 3 + 24 * 3, 0.0);

	return BfmStatus_Ok;
}


void HeadPoseEstimationProblem::dlt()
{
	BFM_DEBUG("Calculating extrinsic parameters using Direct Linear Transform.\n");
	unsigned int nLandmarks = m_pModel->getNLandmarks();
	assert(nLandmarks >= 6);

	cv::Ptr<CvMat> matL;
	double* L;
	double LL[12 * 12], LW[12], LV[12 * 12];
	CvMat _LL = cvMat(12, 12, CV_64F, LL);
	CvMat _LW = cvMat(12, 1, CV_64F, LW);
	CvMat _LV = cvMat(12, 12, CV_64F, LV);
	CvMat _RRt, _RR, _tt;

	matL.reset(cvCreateMat(2 * nLandmarks, 12, CV_64F));
	L = matL->data.db;
	const VectorXd &vecLandmarkBlendshape = m_pModel->getLandmarkCurrentBlendshape();

	const double fx = m_pModel->getFx();
	const double fy = m_pModel->getFy();
	const double cx = m_pModel->getCx();
	const double cy = m_pModel->getCy();

	for(int i = 0; i < N_LANDMARKS; i++, L += 24)
	{
		double x = m_pObservedPoints->part(i).x(), y = m_pObservedPoints->part(i).y();
		double X = vecLandmarkBlendshape(i * 3);
		double Y = vecLandmarkBlendshape(i * 3 + 1);
		double Z = vecLandmarkBlendshape(i * 3 + 2);
		L[0] = X * fx; L[16] = X * fy;
		L[1] = Y * fx; L[17] = Y * fy;
		L[2] = Z * fx; L[18] = Z * fy;
		L[3] = fx; L[19] = fy;
		L[4] = L[5] = L[6] = L[7] = 0.;
		L[12] = L[13] = L[14] = L[15] = 0.;
		L[8] = X * cx - x * X;
		L[9] = Y * cx - x * Y;
		L[10] = Z * cx - x * Z;
		L[11] = cx - x;
		L[20] = X * cy - y * X;
		L[21] = Y * cy - y * Y;
		L[22] = Z * cy - y * Z;
		L[23] = cy - y;
	}

	cvMulTransposed(matL, &_LL, 1);
	cvSVD(&_LL, &_LW, 0, &_LV, CV_SVD_MODIFY_A + CV_SVD_V_T);
	_RRt = cvMat(3, 4, CV_64F, LV + 11 * 12);
	cvGetCols(&_RRt, &_RR, 0, 3);
	cvGetCol(&_RRt, &_tt, 3);

	m_pModel->setMatR(&_RR);
	m_pModel->setVecT(&_tt);
	m_pModel->genExtParams();

}


void HeadPoseEstimationProblem::solve()
{
	enum class RotateType
	{
		Invalid, CW, CCW,
	};

	const unsigned int START = 0;
	const unsigned int END = 20;
	assert(START >= 0 && END <= N_PHOTOS);
	const double EST_THESHOLD = 5e5;

	std::vector<dlib::array2d<dlib::rgb_pixel>> aArr2dImg(N_PHOTOS), aArr2dTransImg(N_PHOTOS);
	const std::vector<Texture>& aTextures = m_pDataManager->getTextures();
	RotateType *pRotateList = new RotateType[N_PHOTOS]{
		RotateType::CCW, RotateType::CCW, RotateType::CCW, RotateType::CCW, RotateType::CCW, 
		RotateType::CW, RotateType::CW, RotateType::CW, RotateType::CW, RotateType::CW,
		RotateType::CCW, RotateType::CCW, RotateType::CCW, RotateType::CCW, RotateType::CCW,
		RotateType::CCW, RotateType::CCW, RotateType::CCW, RotateType::CCW, RotateType::CW,
		RotateType::CW, RotateType::CW, RotateType::CW, RotateType::CW
	};
	
	std::vector<dlib::full_object_detection> aDetRecs(N_PHOTOS), aTransDetRecs(N_PHOTOS);
	std::vector<std::pair<dlib::full_object_detection, int>> aObjDets;
	const unsigned int N_TOTAL_EXT_PARAMS = 3 + 3 * N_PHOTOS;
	double *pExtParams = new double[N_TOTAL_EXT_PARAMS];

	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor sp;
	dlib::deserialize(DLIB_LANDMARK_DETECTOR_DATA_PATH) >> sp;
	BFM_DEBUG(PRINT_YELLOW "\n[Step 0] Detector init successfully\n" COLOR_END);

	BFM_DEBUG(PRINT_YELLOW "\n[Input] %d Images (Id from %d to %d)\n" COLOR_END, END - START, START, END - 1);
	BFM_DEBUG(PRINT_YELLOW "\n[Step 1] Remove outliers\n" COLOR_END);
	for(auto i = START; i < END; i++)
	{
		std::fill(pExtParams, pExtParams + N_TOTAL_EXT_PARAMS, 0.0);
		const Texture& tex = aTextures[i];
		const std::string &strTexPath = tex.getPath();
		std::cout << "Detect " << strTexPath << std::endl;
		load_image(aArr2dImg[i], strTexPath);
		dlib::resize_image(ZOOM_SCALE, aArr2dImg[i]);
		std::cout << "	Now size of image: " << aArr2dImg[i].nr() << " " << aArr2dImg[i].nc() << std::endl;

		if(pRotateList[i] == RotateType::CCW)
			dlib::rotate_image(aArr2dImg[i], aArr2dTransImg[i], M_PI * 0.5);
		else if(pRotateList[i] == RotateType::CW)
			dlib::rotate_image(aArr2dImg[i], aArr2dTransImg[i], M_PI * 1.5);
		else
			continue;
	
		std::vector<dlib::rectangle> aDets = detector(aArr2dTransImg[i]);
		if(aDets.size() == 0)
		{
			std::cout << "Not Found" << std::endl;
			pRotateList[i] = RotateType::Invalid;
			continue;
		}
		dlib::full_object_detection objDetection = sp(aArr2dTransImg[i], aDets[0]);
		aDetRecs[i] = objDetection;

		for(auto j = 0; j < N_LANDMARKS; j++)
		{
			int x = objDetection.part(j).x(), y = objDetection.part(j).y();
			if(pRotateList[i] == RotateType::CCW)
			{
				objDetection.part(j).x() = tex.getWidth() * ZOOM_SCALE - y;
				objDetection.part(j).y() = x;
			}
			else
			{
				objDetection.part(j).x() = y;
				objDetection.part(j).y() = tex.getHeight() * ZOOM_SCALE - x;
			}
		}

		aTransDetRecs[i] = objDetection;
		aObjDets.push_back(std::pair<dlib::full_object_detection, int>(objDetection, i));
		
		ceres::Solver::Summary sum = this->estExtParams(aObjDets, pExtParams);
		if(sum.final_cost > EST_THESHOLD)
		{
			BFM_DEBUG("[NOTE] %d Image is seen as outliers.\n", i);
			pRotateList[i] = RotateType::Invalid;
		}

		aObjDets.clear();
	}

	{
		std::fill(pExtParams, pExtParams + N_TOTAL_EXT_PARAMS, 0.0);
		const unsigned int I_INIT_PHOTO = 4;
		assert(pRotateList[I_INIT_PHOTO] != RotateType::Invalid);
		dlib::image_window initWindow;
		initWindow.clear_overlay();
		initWindow.set_image(aArr2dImg[I_INIT_PHOTO]);
		initWindow.add_overlay(render_face_detections(aTransDetRecs[I_INIT_PHOTO]));
		BFM_DEBUG(PRINT_YELLOW "\n[Step 2] Init extrinsic parameters with random available image (id %d)\n" COLOR_END, I_INIT_PHOTO);
		aObjDets.push_back(std::pair<dlib::full_object_detection, int>(aTransDetRecs[I_INIT_PHOTO], I_INIT_PHOTO));
		this->estExtParams(aObjDets, pExtParams);
		aObjDets.clear();

		for(auto i = 0; i < N_TOTAL_EXT_PARAMS; i++)
			std::cout << pExtParams[i] << " ";
		std::cout << std::endl;

		Eigen::VectorXf vecPts = m_pModel->getLandmarkCurrentBlendshape().template cast<float>();
		float pTmpExtParams[6] = {
			(float)pExtParams[0], (float)pExtParams[1], (float)pExtParams[2],
			(float)pExtParams[3 + I_INIT_PHOTO * 3], (float)pExtParams[3 + I_INIT_PHOTO * 3 + 1], (float)pExtParams[3 + I_INIT_PHOTO * 3 + 2]
		};
		Eigen::VectorXf vecTranPts0 = bfm_utils::TransPoints(pTmpExtParams, vecPts);
		Eigen::VectorXf vecTranPts1 = bfm_utils::TransPoints(m_pDataManager->getCameraMatrices()[I_INIT_PHOTO], vecTranPts0);

		std::vector<dlib::point> aPoints;
		for(auto iLandmark = 0; iLandmark < N_LANDMARKS; iLandmark++) 
		{
			int u = int(m_pModel->getFx() * vecTranPts1(iLandmark * 3) / vecTranPts1(iLandmark * 3 + 2) + m_pModel->getCx());
			int v = int(m_pModel->getFy() * vecTranPts1(iLandmark * 3 + 1) / vecTranPts1(iLandmark * 3 + 2) + m_pModel->getCy());
			aPoints.push_back(dlib::point(u, v));
		}
		initWindow.add_overlay(render_face_detections(dlib::full_object_detection(dlib::rectangle(), aPoints), dlib::rgb_pixel(0, 0, 255)));
		std::cin.get();
		initWindow.close_window();
	}

////////////////////////////////////////////////
	// std::fill(m_aExtParams, m_aExtParams + 3 + 24 * 3, 0.0);
	// for(unsigned int iTex = 4; iTex < 10; iTex++)
	// {
	// 	dlib::image_window win3;
	// 	win3.clear_overlay();
	// 	Texture tex = aTextures[iTex];
	// 	const std::string &strTexPath = tex.getPath();
	// 	std::cout << "Detect " << strTexPath << std::endl;
	// 	dlib::array2d<dlib::rgb_pixel> arr2dImg, arr2dImgTrans;
	// 	load_image(arr2dImg, strTexPath);
	// 	dlib::resize_image(0.1, arr2dImg);
	// 	win3.set_image(arr2dImg);

	// 	std::cout << "	Now size of image: " << arr2dImg.nr() << " " << arr2dImg.nc() << std::endl;

	// 	if(rotateList[iTex] == 0)
	// 	{
	// 		dlib::rotate_image(arr2dImg, arr2dImgTrans, M_PI * 0.5);
	// 	}
	// 	else
	// 	{
	// 		dlib::rotate_image(arr2dImg, arr2dImgTrans, M_PI * 1.5);
	// 	}

	// 	std::vector<dlib::rectangle> aDets = detector(arr2dImgTrans);
	// 	if(aDets.size() == 0)
	// 	{
	// 		std::cout << "Not Found" << std::endl;
	// 		rotateList[iTex] = -1;
	// 		continue;
	// 	}
	// 	dlib::full_object_detection objDetection = sp(arr2dImgTrans, aDets[0]);

	// 	for(auto i = 0; i < N_LANDMARKS; i++)
	// 	{
	// 		int x = objDetection.part(i).x(), y = objDetection.part(i).y();
	// 		if(rotateList[iTex] == 0)
	// 		{
	// 			objDetection.part(i).x() = tex.getWidth() * 0.1 - y;
	// 			objDetection.part(i).y() = x;
	// 		}
	// 		else
	// 		{
	// 			objDetection.part(i).x() = y;
	// 			objDetection.part(i).y() = tex.getHeight() * 0.1 - x;
	// 		}
	// 	}
	// 	win3.add_overlay(render_face_detections(objDetection));
	// 	m_aObjDetections.push_back(std::pair<dlib::full_object_detection, int>(objDetection, iTex));
		
	// 	this->estExtParams();
	// 	this->m_pModel->printExtParams();

	// 	// Eigen::VectorXf vecLandmark = m_pModel->getLandmarkCurrentBlendshapeTransformed().template cast<float>();
	// 	// Eigen::VectorXf vecLandmarkTransformed = bfm_utils::TransPoints(m_pDataManager->getCameraMatrices()[iTex], vecLandmark);
	// 	// // Eigen::VectorXf vecLandmarkTransformed = vecLandmark;

	// 	Eigen::VectorXf vecPts = m_pModel->getLandmarkCurrentBlendshape().template cast<float>();
	// 	float aExtParams[6] = {
	// 		(float)m_aExtParams[0], (float)m_aExtParams[1], (float)m_aExtParams[2],
	// 		(float)m_aExtParams[3 + iTex * 3], (float)m_aExtParams[3 + iTex * 3 + 1], (float)m_aExtParams[3 + iTex * 3 + 2]
	// 	};
	// 	Eigen::VectorXf vecTranPts0 = bfm_utils::TransPoints(aExtParams, vecPts);
	// 	Eigen::VectorXf vecTranPts1 = bfm_utils::TransPoints(m_pDataManager->getCameraMatrices()[iTex], vecTranPts0);

	// 	std::vector<dlib::point> aPoints;

	// 	for(unsigned int iLandmark = 0; iLandmark < N_LANDMARKS; iLandmark++) 
	// 	{
	// 		int u = int(m_pModel->getFx() * vecTranPts1(iLandmark * 3) / vecTranPts1(iLandmark * 3 + 2) + m_pModel->getCx());
	// 		int v = int(m_pModel->getFy() * vecTranPts1(iLandmark * 3 + 1) / vecTranPts1(iLandmark * 3 + 2) + m_pModel->getCy());
	// 		aPoints.push_back(dlib::point(u, v));
	// 	}
	// 	win3.add_overlay(render_face_detections(dlib::full_object_detection(dlib::rectangle(), aPoints), dlib::rgb_pixel(0,0,255)));

	// 	m_pModel->writePly("test" + std::to_string(iTex) + ".ply", ModelWriteMode_CameraCoord);

	// 	// m_pModel->clrExtParams();
	// 	std::fill(m_aExtParams, m_aExtParams + 3 + 24 * 3, 0.0);
	// 	m_aObjDetections.clear();

	// 	std::cin.get();
	// }

	// return;

///////////////////////////////////////////////

	BFM_DEBUG(PRINT_YELLOW "\n[Step 3] Combine available photos and estimate.\n" COLOR_END);
	dlib::image_window winOrigin[END], winTrans[END];
	for(unsigned int i = START; i < END; i++)
	{
		winOrigin[i].clear_overlay();
		winOrigin[i].set_title("Origin-" + std::to_string(i));
		winTrans[i].clear_overlay();
		winTrans[i].set_title("Trans-" + std::to_string(i));

		if(pRotateList[i] == RotateType::Invalid)
		{
			winOrigin[i].close_window();
			winTrans[i].close_window();
			continue;
		}

		winOrigin[i].set_image(aArr2dImg[i]);
		winTrans[i].set_image(aArr2dTransImg[i]);
		winTrans[i].add_overlay(render_face_detections(aDetRecs[i]));
		winOrigin[i].add_overlay(render_face_detections(aTransDetRecs[i]));
		aObjDets.push_back(std::pair<dlib::full_object_detection, int>(aTransDetRecs[i], i));
	}

	std::cout << "Final picked photos include: \n   " << std::endl;
	for(auto itPhoto = aObjDets.cbegin(); itPhoto != aObjDets.cend(); itPhoto++)
		std::cout << " " << itPhoto->second;
	std::cout << std::endl;

	this->estExtParams(aObjDets, pExtParams);
//	this->estShapeCoef();
//	this->estExprCoef();

//	m_pModel->printShapeCoef();
//	m_pModel->printExprCoef();
//	m_pModel->genFace();
//	m_pModel->writePly("test-final.ply", ModelWriteMode_CameraCoord);

	for(int i = 0; i < N_TOTAL_EXT_PARAMS; i++)
		std::cout << m_aExtParams[i] << " ";
	std::cout << std::endl;

	BFM_DEBUG(PRINT_YELLOW "\n[Step 4] Show results.\n" COLOR_END);
	Eigen::VectorXf vecPts = m_pModel->getLandmarkCurrentBlendshape().template cast<float>();
	for(auto i = START; i < END; i++)
	{	
		if(pRotateList[i] == RotateType::Invalid) continue;

		float pTmpExtParams[6] = {
			(float)pExtParams[0], (float)pExtParams[1], (float)pExtParams[2],
			(float)pExtParams[3 + i * 3], (float)pExtParams[3 + i * 3 + 1], (float)pExtParams[3 + i * 3 + 2]
		};
		Eigen::VectorXf vecTranPts0 = bfm_utils::TransPoints(pTmpExtParams, vecPts);
		Eigen::VectorXf vecTranPts1 = bfm_utils::TransPoints(m_pDataManager->getCameraMatrices()[i], vecTranPts0);
		
		std::vector<dlib::point> aPoints;

		for(unsigned int iLandmark = 0; iLandmark < N_LANDMARKS; iLandmark++) 
		{
			int u = int(m_pModel->getFx() * vecTranPts1(iLandmark * 3) / vecTranPts1(iLandmark * 3 + 2) + m_pModel->getCx());
			int v = int(m_pModel->getFy() * vecTranPts1(iLandmark * 3 + 1) / vecTranPts1(iLandmark * 3 + 2) + m_pModel->getCy());
			aPoints.push_back(dlib::point(u, v));
			// std::cout << u << " " << v << std::endl;
		}
		winOrigin[i].add_overlay(render_face_detections(dlib::full_object_detection(dlib::rectangle(), aPoints), dlib::rgb_pixel(0, 0, 255)));
	}

	aObjDets.clear();
	delete[] pRotateList;
	delete[] pExtParams;
	std::cin.get();
}


ceres::Solver::Summary HeadPoseEstimationProblem::estExtParams(const DetPairVector&  aObjDets, double *pExtParams)
{
	BFM_DEBUG(PRINT_GREEN "#################### Estimate Multi-Faces Extrinsic Parameters ####################\n" COLOR_END);
	CERES_INIT(N_CERES_ITERATIONS, N_CERES_THREADS, B_CERES_STDCOUT);
	ceres::CostFunction *costFunction = MultiExtParamsReprojErr::create(aObjDets, m_pModel, m_pDataManager);
	ceres::Problem problem;
	problem.AddResidualBlock(costFunction, nullptr, pExtParams);
	ceres::Solve(options, &problem, &summary);
	BFM_DEBUG("%s\n", summary.BriefReport().c_str());
	// m_pModel->genTransMat();
	return summary;
}

void HeadPoseEstimationProblem::estShapeCoef() 
{
	BFM_DEBUG(PRINT_GREEN "#################### Estimate Shape Coefficients ####################\n" COLOR_END);
	ceres::Problem problem;
	double *aShapeCoef = m_pModel->getMutableShapeCoef();
	ceres::CostFunction *costFunction = MultiShapeCoefReprojErr::create(m_aObjDetections, m_pModel, m_pDataManager, m_aExtParams);
	problem.AddResidualBlock(costFunction, nullptr, aShapeCoef);
	CERES_INIT(N_CERES_ITERATIONS, N_CERES_THREADS, B_CERES_STDCOUT);
	ceres::Solve(options, &problem, &summary);
	BFM_DEBUG("%s\n", summary.BriefReport().c_str());
	m_pModel->genLandmarkBlendshape();
}


void HeadPoseEstimationProblem::estExprCoef() 
{
	BFM_DEBUG(PRINT_GREEN "#################### Estimate Expression Coefficients ####################\n" COLOR_END);
	ceres::Problem problem;
	double *aExprCoef = m_pModel->getMutableExprCoef();
	ceres::CostFunction *costFunction = MultiExprCoefReprojErr::create(m_aObjDetections, m_pModel, m_pDataManager, m_aExtParams);
	problem.AddResidualBlock(costFunction, nullptr, aExprCoef);
	CERES_INIT(N_CERES_ITERATIONS, N_CERES_THREADS, B_CERES_STDCOUT);
	ceres::Solve(options, &problem, &summary);
	BFM_DEBUG("%s\n", summary.BriefReport().c_str());
	m_pModel->genLandmarkBlendshape();
}


bool HeadPoseEstimationProblem::solveExtParams(long mode, double ca, double cb) 
{
	BFM_DEBUG(PRINT_GREEN "#################### Estimate Extrinsic Parameters ####################\n" COLOR_END);
	if(mode & SolveExtParamsMode_UseOpenCV)
	{
		const double *aIntParams = m_pModel->getIntParams();
		double aCameraMatrix[3][3] = {
			{aIntParams[0], 0.0, aIntParams[2]},
			{0.0, aIntParams[1], aIntParams[3]},
			{0.0, 0.0, 1.0}
		};
		cv::Mat matCameraMatrix = cv::Mat(3, 3, CV_64FC1, aCameraMatrix);		
		#ifdef _DEBUG
		std::cout << "camera matrix: " << matCameraMatrix << std::endl;
		#endif
		std::vector<float> aDistCoef(0);
		std::vector<cv::Point3f> out;
		std::vector<cv::Point2f> in;
		cv::Mat rvec, tvec;
		const VectorXd& vecLandmarkBlendshape = m_pModel->getLandmarkCurrentBlendshape();
		for(unsigned int iLandmark = 0; iLandmark < 68; iLandmark++) {
			out.push_back(cv::Point3f(
				vecLandmarkBlendshape(iLandmark * 3), 
				vecLandmarkBlendshape(iLandmark * 3 + 1), 
				vecLandmarkBlendshape(iLandmark * 3 + 2)));
			in.push_back(cv::Point2f(m_pObservedPoints->part(iLandmark).x(), m_pObservedPoints->part(iLandmark).y()));
		}
		cv::solvePnP(out, in, matCameraMatrix, aDistCoef, rvec, tvec);
		cv::Rodrigues(rvec, rvec);
		#ifdef _DEBUG
		std::cout << rvec << std::endl;
		std::cout << tvec << std::endl;
		#endif
		m_pModel->setMatR(rvec);
		m_pModel->setVecT(tvec);
		m_pModel->genExtParams();
		return true;
	}
	else if(mode & SolveExtParamsMode_UseLinearizedRadians)
	{
		BFM_DEBUG("solve -> external parameters (linealized)\n");
		if(mode & SolveExtParamsMode_UseDlt)
		{
			BFM_DEBUG("	1) estimate initial values by using DLT algorithm.\n");
			dlt();
		}
		else
		{
			BFM_DEBUG("	1) initial values have been set in advance or are 0s.\n");
		}
		
		m_pModel->genTransMat();	

		CERES_INIT(N_CERES_ITERATIONS, N_CERES_THREADS, B_CERES_STDCOUT);
		while(true) 
		{
			ceres::Problem problem;
			double aSmallExtParams[6] = { 0.f };
			ceres::CostFunction *costFunction = LinearizedExtParamsReprojErr::create(m_pObservedPoints, m_pModel, m_aLandmarkMap, ca, cb);
			problem.AddResidualBlock(costFunction, nullptr, aSmallExtParams);
			ceres::Solve(options, &problem, &summary);
			BFM_DEBUG("%s\n", summary.BriefReport().c_str());
			
			if(is_close_enough(aSmallExtParams, 0, 0)) 
			{
				#ifdef _DEBUG
				bfm_utils::PrintArr(aSmallExtParams, 6);
				std::cout << summary.BriefReport() << std::endl;
				#endif
				break; 
			}

			m_pModel->accExtParams(aSmallExtParams);
			
			#ifdef _DEBUG
			bfm_utils::PrintArr(aSmallExtParams, 6);							
			#endif
		}
		m_pModel->genExtParams();
		return (summary.termination_type == ceres::CONVERGENCE);
	}
	else
	{
		#ifndef HPE_SHUT_UP
		std::cout << "solve -> external parameters" << std::endl;	
		std::cout << "init ceres solve - ";
		#endif

		#ifndef HPE_SHUT_UP
		if(mode & SolveExtParamsMode_UseDlt)
		{
			std::cout << "	1) esitimate initial values by using DLT algorithm." << std::endl;
			dlt();
		}
		else
		{
			std::cout << "	1) initial values have been set in advance or are 0s." << std::endl;
		}
		#else
		if(mode & SolveExtParamsMode_UseDlt) dlt();
		#endif

		ceres::Problem problem;
		double *ext_params = m_pModel->getMutableExtParams();
		ceres::CostFunction *costFunction = ExtParamsReprojErr::create(m_pObservedPoints, m_pModel, m_aLandmarkMap);
		problem.AddResidualBlock(costFunction, nullptr, ext_params);
		ceres::Solver::Options options;
		options.max_num_iterations = 100;
		options.num_threads = 8;
		options.minimizer_progress_to_stdout = true;
		ceres::Solver::Summary summary;
		#ifndef HPE_SHUT_UP
		std::cout << "success" << std::endl;
		#endif
		ceres::Solve(options, &problem, &summary);
		#ifndef HPE_SHUT_UP
		std::cout << summary.BriefReport() << std::endl;
		#endif
		m_pModel->genTransMat();
		return (summary.termination_type == ceres::CONVERGENCE);
	}
}


bool HeadPoseEstimationProblem::solveShapeCoef() {
	BFM_DEBUG(PRINT_GREEN "#################### Estimate Shape Coefficients ####################\n" COLOR_END);
	ceres::Problem problem;
	double *aShapeCoef = m_pModel->getMutableShapeCoef();
	ceres::CostFunction *costFunction = ShapeCoefReprojErr::create(m_pObservedPoints, m_pModel, m_aLandmarkMap);
	ceres::DynamicAutoDiffCostFunction<ShapeCoefRegTerm> *regTerm = ShapeCoefRegTerm::create(m_pModel);
	regTerm->AddParameterBlock(m_pModel->getNIdPcs());
	regTerm->SetNumResiduals(m_pModel->getNIdPcs());
	problem.AddResidualBlock(costFunction, nullptr, aShapeCoef);
	problem.AddResidualBlock(regTerm, nullptr, aShapeCoef);
	CERES_INIT(N_CERES_ITERATIONS, N_CERES_THREADS, B_CERES_STDCOUT);
	ceres::Solve(options, &problem, &summary);
	BFM_DEBUG("%s\n", summary.BriefReport().c_str());
	m_pModel->genLandmarkBlendshape();
	return (summary.termination_type == ceres::CONVERGENCE);
}


bool HeadPoseEstimationProblem::solveExprCoef() {
	BFM_DEBUG(PRINT_GREEN "#################### Estimate Expression Coefficients ####################\n" COLOR_END);
	ceres::Problem problem;
	double *aExprCoef = m_pModel->getMutableExprCoef();
	ceres::CostFunction *costFunction = ExprCoefReprojErr::create(m_pObservedPoints, m_pModel, m_aLandmarkMap);
	ceres::DynamicAutoDiffCostFunction<ExprCoefRegTerm> *regTerm = ExprCoefRegTerm::create(m_pModel);
	regTerm->AddParameterBlock(m_pModel->getNExprPcs());
	regTerm->SetNumResiduals(m_pModel->getNExprPcs());
	problem.AddResidualBlock(costFunction, nullptr, aExprCoef);
	problem.AddResidualBlock(regTerm, nullptr, aExprCoef);
	CERES_INIT(N_CERES_ITERATIONS, N_CERES_THREADS, B_CERES_STDCOUT);
	ceres::Solve(options, &problem, &summary);
	BFM_DEBUG("%s\n", summary.BriefReport().c_str());
	m_pModel->genLandmarkBlendshape();
	return (summary.termination_type == ceres::CONVERGENCE);
}


bool HeadPoseEstimationProblem::is_close_enough(double *ext_params, double rotation_eps, double translation_eps)
{
	for(int i=0; i<3; i++)
		if(abs(ext_params[i]) > rotation_eps)
			return false;
	for(int i=3; i<6; i++)
		if(abs(ext_params[i]) > translation_eps)
			return false;
	return true;
}


