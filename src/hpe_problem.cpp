#include "hpe_problem.h"

MHPEProblem::MHPEProblem(
	const std::string& strProjectPath,
	const std::string& strBfmH5Path,
	const std::string& strLandmarkIdxPath
) 
{
	if(this->init(strProjectPath, strBfmH5Path, strLandmarkIdxPath) == Status_Error)
		LOG(ERROR) << "Multi-view head pose estimation problem structure init failed.";
}

Status MHPEProblem::init(
	const std::string& strProjectPath,
	const std::string& strBfmH5Path,
	const std::string& strLandmarkIdxPath
) 
{
	LOG(INFO) << "Init problem structure of multiple head pose estimation.";

	m_pDataManager = new DataManager(strProjectPath);
	if(m_pDataManager->getFaces() == 0)
	{
		LOG(ERROR) << "Load data manager failed.";
		return Status_Error;
	}

	std::array<double, N_INT_PARAMS> aIntParams = { };
	aIntParams[0] = aIntParams[1] = m_pDataManager->getF() * ZOOM_SCALE;
	aIntParams[2] = (m_pDataManager->getWidth() * 0.5 + m_pDataManager->getCx()) * ZOOM_SCALE;
	aIntParams[3] = (m_pDataManager->getHeight() * 0.5 + m_pDataManager->getCy()) * ZOOM_SCALE;
	
	m_pBfmManager = new BfmManager(
		strBfmH5Path,
		aIntParams,
		strLandmarkIdxPath
	);
	if(m_pBfmManager->getStrModelPath() == "")
	{
		LOG(ERROR) << "Load BFM failed.";
		return Status_Error;
	}
	m_pBfmManager->check();

	return Status_Ok;
}


void MHPEProblem::solve(SolveExtParamsMode mode, const std::string& strDlibDetPath)
{
	LOG(INFO) << "Begin solving multi-view head pose estimation.";

	std::vector<dlib::array2d<dlib::rgb_pixel>> aArr2dImg(N_PHOTOS), aArr2dTransImg(N_PHOTOS);
	const std::vector<Texture>& aTextures = m_pDataManager->getTextures();

	RotateType *pRotateList = new RotateType[N_PHOTOS]{	// Outlier: 2(DETECT TOO BAD) 3(FIT TOO BAD) 20(FIT NOT GOOD)
		RotateType_Invalid, RotateType_Invalid, RotateType_Invalid, RotateType_Invalid, RotateType_CCW, 
		RotateType_CW, RotateType_CW, RotateType_CW, RotateType_CW, RotateType_Invalid,
		RotateType_CCW, RotateType_Invalid, RotateType_CCW, RotateType_Invalid, RotateType_CCW,
		RotateType_CCW, RotateType_CCW, RotateType_CCW, RotateType_CCW, RotateType_Invalid,
		RotateType_Invalid, RotateType_CW, RotateType_Invalid, RotateType_Invalid
	};
	
	std::vector<dlib::full_object_detection> aDetRecs(N_PHOTOS), aTransDetRecs(N_PHOTOS);
	std::vector<std::pair<dlib::full_object_detection, int>> aObjDets;

	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor sp;
	dlib::deserialize(strDlibDetPath) >> sp;
	LOG(INFO) << "[Step 0] Detector init successfully";
	LOG(INFO) << "\t[Input] " <<  END - START << "Images (Id from " << START << " to " << END - 1 << ".";
	LOG(INFO) << "[Step 1] Remove outliers";
	for(auto i = START; i < END; i++)
	{
		if(pRotateList[i] == RotateType_Invalid)
			continue;

		const Texture& tex = aTextures[i];
		const std::string &strTexPath = tex.getPath();
		std::cout << "Detect " << strTexPath << std::endl;
		load_image(aArr2dImg[i], strTexPath);
		dlib::resize_image(ZOOM_SCALE, aArr2dImg[i]);
		std::cout << "	Now size of image: " << aArr2dImg[i].nr() << " " << aArr2dImg[i].nc() << std::endl;

		if(pRotateList[i] == RotateType_CCW)
			dlib::rotate_image(aArr2dImg[i], aArr2dTransImg[i], M_PI * 0.5);
		else if(pRotateList[i] == RotateType_CW)
			dlib::rotate_image(aArr2dImg[i], aArr2dTransImg[i], M_PI * 1.5);
		else
			continue;
	
		std::vector<dlib::rectangle> aDets = detector(aArr2dTransImg[i]);
		if(aDets.size() == 0)
		{
			std::cout << "Not Found" << std::endl;
			pRotateList[i] = RotateType_Invalid;
			continue;
		}
		dlib::full_object_detection objDetection = sp(aArr2dTransImg[i], aDets[0]);
		aDetRecs[i] = objDetection;

		for(auto j = 0; j < N_LANDMARKS; j++)
		{
			int x = objDetection.part(j).x(), y = objDetection.part(j).y();
			if(pRotateList[i] == RotateType_CCW)
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
		
		// ceres::Solver::Summary sum = this->estExtParams(aObjDets, pExtParams);
		// if(sum.final_cost > EST_THESHOLD)
		// {
			// BFM_DEBUG("[NOTE] %d Image is seen as outliers.\n", i);
			// pRotateList[i] = RotateType_Invalid;
		// }

		aObjDets.clear();
	}


	LOG(INFO) << "[Step 3] Combine available photos and estimate.";
	
	dlib::image_window winOrigin[END], winTrans[END];
	for(auto i = 0u; i < START; i++) 
	{
		winOrigin[i].close_window();
		winTrans[i].close_window();
	}

	
	for(unsigned int i = START; i < END; i++)
	{
		if(pRotateList[i] == RotateType_Invalid)
		{
			winOrigin[i].close_window();
			winTrans[i].close_window();
			continue;
		}

		winOrigin[i].clear_overlay();
		winOrigin[i].set_title("Origin-" + std::to_string(i));
		winTrans[i].clear_overlay();
		winTrans[i].set_title("Trans-" + std::to_string(i));

		winOrigin[i].set_image(aArr2dImg[i]);
		winTrans[i].set_image(aArr2dTransImg[i]);
		winTrans[i].add_overlay(render_face_detections(aDetRecs[i]));
		winOrigin[i].add_overlay(render_face_detections(aTransDetRecs[i]));
		aObjDets.push_back(std::pair<dlib::full_object_detection, int>(aTransDetRecs[i], i));
	}

	{
		auto&& log = COMPACT_GOOGLE_LOG_INFO;
		LOG(INFO) << "Final picked photos include:";
		for(auto itPhoto = aObjDets.cbegin(); itPhoto != aObjDets.cend(); itPhoto++)
			log.stream() << " " << itPhoto->second;
		log.stream() << "\n";            
	}

	ceres::Problem problem;	
	ceres::CostFunction *costFunction;
	CERES_INIT(N_CERES_ITERATIONS, N_CERES_THREADS, B_CERES_STDCOUT);
	costFunction = PointReprojErr::create(aObjDets, m_pBfmManager, m_pDataManager);
	double *points = new double[204];
	std::fill(points, points + 204, 0.0);
	problem.AddResidualBlock(costFunction, nullptr, points);
	ceres::Solve(options, &problem, &summary);
	
	int idxPairHor[2] = {27, 8};
	int idxPairVer[2] = {0, 16};
	double x1, y1, z1, x2, y2, z2;
	double dis1, dis2;
	double scHor, scVer, initSc;
	auto& modelPoints = m_pBfmManager->getLandmarkCurrentBlendshape();
	x1 = points[idxPairHor[0] * 3];
	y1 = points[idxPairHor[0] * 3 + 1];
	z1 = points[idxPairHor[0] * 3 + 2];
	x2 = points[idxPairHor[1] * 3];
	y2 = points[idxPairHor[1] * 3 + 1];
	z2 = points[idxPairHor[1] * 3 + 2];
	dis1 = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2);
	x1 = modelPoints(idxPairHor[0] * 3);
	y1 = modelPoints(idxPairHor[0] * 3 + 1);
	z1 = modelPoints(idxPairHor[0] * 3 + 2);
	x2 = modelPoints(idxPairHor[1] * 3);
	y2 = modelPoints(idxPairHor[1] * 3 + 1);
	z2 = modelPoints(idxPairHor[1] * 3 + 2);
	dis2 = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2);
	scHor = sqrt(dis1) / sqrt(dis2);
	x1 = points[idxPairVer[0] * 3];
	y1 = points[idxPairVer[0] * 3 + 1];
	z1 = points[idxPairVer[0] * 3 + 2];
	x2 = points[idxPairVer[1] * 3];
	y2 = points[idxPairVer[1] * 3 + 1];
	z2 = points[idxPairVer[1] * 3 + 2];
	dis1 = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2);
	x1 = modelPoints(idxPairVer[0] * 3);
	y1 = modelPoints(idxPairVer[0] * 3 + 1);
	z1 = modelPoints(idxPairVer[0] * 3 + 2);
	x2 = modelPoints(idxPairVer[1] * 3);
	y2 = modelPoints(idxPairVer[1] * 3 + 1);
	z2 = modelPoints(idxPairVer[1] * 3 + 2);
	dis2 = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2);
	scVer = sqrt(dis1) / sqrt(dis2);	
	initSc = (scHor + scVer) * 0.5;
	LOG(INFO) << "Estimate scale: " << scHor << " " << scVer << " -> " << initSc;

	this->estExtParams(points, initSc);
	bool isConvergence;
	Summary sum;
	do {
		isConvergence = true;
		sum = this->estExtParams(aObjDets);
		isConvergence &= (sum.initial_cost - sum.final_cost < 1e2);
		sum = this->estShapeCoef(aObjDets);
		isConvergence &= (sum.initial_cost - sum.final_cost < 1e2);
		sum = this->estExprCoef(aObjDets);
		isConvergence &= (sum.initial_cost - sum.final_cost < 1e2);
	} while(!isConvergence);
	

	BFM_DEBUG(PRINT_YELLOW "\n[Step 4] Show results.\n" COLOR_END);
	float pfExtParams[N_EXT_PARAMS];
	for(auto i = 0u; i < N_EXT_PARAMS; ++i) pfExtParams[i] = (float)m_pBfmManager->getMutableExtParams()[i];
	Eigen::VectorXf vecPts = m_pBfmManager->getLandmarkCurrentBlendshape().template cast<float>() * (float)m_pBfmManager->getMutableScale();
	// Eigen::VectorXf vecPts = m_pBfmManager->getLandmarkCurrentBlendshape().template cast<float>();
	for(auto i = START; i < END; i++)
	{	
		if(pRotateList[i] == RotateType_Invalid) continue;

		Eigen::VectorXf vecTranPts0 = bfm_utils::TransPoints(pfExtParams, vecPts);
		Eigen::VectorXf vecTranPts1 = bfm_utils::TransPoints(m_pDataManager->getCameraMatrices()[i], vecTranPts0);
		// Eigen::VectorXf vecTranPts1 = vecTranPts0;

		std::vector<dlib::point> aPoints;
		for(unsigned int iLandmark = 0; iLandmark < N_LANDMARKS; iLandmark++) 
		{
			int u = int(m_pBfmManager->getFx() * vecTranPts1(iLandmark * 3) / vecTranPts1(iLandmark * 3 + 2) + m_pBfmManager->getCx());
			int v = int(m_pBfmManager->getFy() * vecTranPts1(iLandmark * 3 + 1) / vecTranPts1(iLandmark * 3 + 2) + m_pBfmManager->getCy());
			aPoints.push_back(dlib::point(u, v));
			// std::cout << u << " " << v << std::endl;
		}
		winOrigin[i].add_overlay(render_face_detections(dlib::full_object_detection(dlib::rectangle(), aPoints), dlib::rgb_pixel(0, 0, 255)));
	}


	// INTIAL
	for(auto i = START; i < END; i++)
	{	
		if(pRotateList[i] == RotateType_Invalid) continue;

		Eigen::VectorXf vecTranPts0;
		vecTranPts0.resize(204);
		for(auto j = 0; j < 204; j++)
			vecTranPts0(j) = points[j];		
		Eigen::VectorXf vecTranPts1 = bfm_utils::TransPoints(m_pDataManager->getCameraMatrices()[i], vecTranPts0);
		// Eigen::VectorXf vecTranPts1 = vecTranPts0;

		std::vector<dlib::point> aPoints;
		for(unsigned int iLandmark = 0; iLandmark < N_LANDMARKS; iLandmark++) 
		{
			int u = int(m_pBfmManager->getFx() * vecTranPts1(iLandmark * 3) / vecTranPts1(iLandmark * 3 + 2) + m_pBfmManager->getCx());
			int v = int(m_pBfmManager->getFy() * vecTranPts1(iLandmark * 3 + 1) / vecTranPts1(iLandmark * 3 + 2) + m_pBfmManager->getCy());
			aPoints.push_back(dlib::point(u, v));
			// std::cout << u << " " << v << std::endl;
		}
		winOrigin[i].add_overlay(render_face_detections(dlib::full_object_detection(dlib::rectangle(), aPoints), dlib::rgb_pixel(0, 255, 255)));
	}


	m_pBfmManager->genFace();
	m_pBfmManager->writePly("face_ext_shape.ply", ModelWriteMode_CameraCoord);
	// m_pBfmManager->writePly("avg.ply");

	aObjDets.clear();
	delete[] pRotateList;
	std::cin.get();
}


void MHPEProblem::estExtParams(double *pPoints, double dScMean)
{
	LOG(INFO) << "Estimate Multi-Faces Extrinsic Parameters (Initial)" << std::endl;

	double& dScale = this->m_pBfmManager->getMutableScale();
	double *pExtParams;
	ceres::Problem problem;	
	ceres::CostFunction *costFunction;
	CERES_INIT(N_CERES_ITERATIONS, N_CERES_THREADS, B_CERES_STDCOUT);
	pExtParams = this->m_pBfmManager->getMutableExtParams().data();
	costFunction = MultiExtParams3D3DReprojErr::create(pPoints, dScMean, m_pBfmManager, m_pDataManager);
	problem.AddResidualBlock(costFunction, nullptr, pExtParams);
	ceres::Solve(options, &problem, &summary);
	
	BFM_DEBUG("%s\n", summary.BriefReport().c_str());
	LOG(INFO) << "Scale: " << dScale << std::endl;
	
	// Sync result
	m_pBfmManager->genTransMat();	
}


Summary MHPEProblem::estExtParams(const DetPairVector& aObjDets)
{
	LOG(INFO) << "Estimate Multi-Faces Extrinsic Parameters" << std::endl;

	double& dScale = this->m_pBfmManager->getMutableScale();
	double *pExtParams;
	ceres::Problem problem;	
	ceres::CostFunction *costFunction;
	CERES_INIT(N_CERES_ITERATIONS, N_CERES_THREADS, B_CERES_STDCOUT);
	
	pExtParams = this->m_pBfmManager->getMutableExtParams().data();
	
	costFunction = MultiExtParamsReprojErr::create(aObjDets, m_pBfmManager, m_pDataManager);
	problem.AddResidualBlock(costFunction, nullptr, pExtParams, &dScale);
	ceres::Solve(options, &problem, &summary);
	
	BFM_DEBUG("%s\n", summary.BriefReport().c_str());
	LOG(INFO) << "Scale: " << dScale << std::endl;
	
	// Sync result
	m_pBfmManager->genTransMat();
	
	return summary;
}

Summary MHPEProblem::estShapeCoef(const DetPairVector& aObjDets) 
{
	LOG(INFO) << "Estimate Multi-Faces Shape Coefficients" << std::endl;

	double *pShapeCoef;
	ceres::Problem problem;
	ceres::CostFunction *costFunction;
	CERES_INIT(N_CERES_ITERATIONS, N_CERES_THREADS, B_CERES_STDCOUT);

	pShapeCoef = m_pBfmManager->getMutableShapeCoef();
	
	costFunction = MultiShapeCoefReprojErr::create(aObjDets, m_pBfmManager, m_pDataManager);
	problem.AddResidualBlock(costFunction, nullptr, pShapeCoef);
	ceres::Solve(options, &problem, &summary);

	LOG(INFO) << summary.BriefReport() << std::endl;
 
	// Sync	result
	m_pBfmManager->genLandmarkBlendshape();

	return summary;
}


Summary MHPEProblem::estExprCoef(const DetPairVector& aObjDets) 
{
	LOG(INFO) << "Estimate Multi-Faces Expression Coefficients" << std::endl;

	double *pExprCoef;
	ceres::Problem problem;
	ceres::CostFunction *costFunction;
	CERES_INIT(N_CERES_ITERATIONS, N_CERES_THREADS, B_CERES_STDCOUT);
	
	pExprCoef = m_pBfmManager->getMutableExprCoef();

	costFunction = MultiExprCoefReprojErr::create(aObjDets, m_pBfmManager, m_pDataManager);
	problem.AddResidualBlock(costFunction, nullptr, pExprCoef);
	ceres::Solve(options, &problem, &summary);

	LOG(INFO) << summary.BriefReport() << std::endl;

	// Sync	result
	m_pBfmManager->genLandmarkBlendshape();

	return summary;
}


bool MHPEProblem::solveExtParams(long mode, double ca, double cb) 
{
	BFM_DEBUG(PRINT_GREEN "#################### Estimate Extrinsic Parameters ####################\n" COLOR_END);
	if(mode & SolveExtParamsMode_UseOpenCV)
	{
		const double *aIntParams = m_pBfmManager->getIntParams().data();
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
		const VectorXd& vecLandmarkBlendshape = m_pBfmManager->getLandmarkCurrentBlendshape();
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
		m_pBfmManager->setMatR(rvec);
		m_pBfmManager->setVecT(tvec);
		m_pBfmManager->genExtParams();
		return true;
	}
	else if(mode & SolveExtParamsMode_UseLinearizedRadians)
	{
		BFM_DEBUG("solve -> external parameters (linealized)\n");
		if(mode & SolveExtParamsMode_UseDlt)
		{
			BFM_DEBUG("	1) estimate initial values by using DLT algorithm.\n");
			// dlt();
		}
		else
		{
			BFM_DEBUG("	1) initial values have been set in advance or are 0s.\n");
		}
		
		m_pBfmManager->genTransMat();	

		CERES_INIT(N_CERES_ITERATIONS, N_CERES_THREADS, B_CERES_STDCOUT);
		while(true) 
		{
			ceres::Problem problem;
			double aSmallExtParams[6] = { 0.f };
			ceres::CostFunction *costFunction = LinearizedExtParamsReprojErr::create(m_pObservedPoints, m_pBfmManager, m_aLandmarkMap, ca, cb);
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

			m_pBfmManager->accExtParams(aSmallExtParams);
			
			#ifdef _DEBUG
			bfm_utils::PrintArr(aSmallExtParams, 6);							
			#endif
		}
		m_pBfmManager->genExtParams();
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
			// dlt();
		}
		else
		{
			std::cout << "	1) initial values have been set in advance or are 0s." << std::endl;
		}
		#else
		if(mode & SolveExtParamsMode_UseDlt) dlt();
		#endif

		ceres::Problem problem;
		double *ext_params = m_pBfmManager->getMutableExtParams().data();
		ceres::CostFunction *costFunction = ExtParamsReprojErr::create(m_pObservedPoints, m_pBfmManager, m_aLandmarkMap);
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
		m_pBfmManager->genTransMat();
		return (summary.termination_type == ceres::CONVERGENCE);
	}
}


bool MHPEProblem::solveShapeCoef() {
	BFM_DEBUG(PRINT_GREEN "#################### Estimate Shape Coefficients ####################\n" COLOR_END);
	ceres::Problem problem;
	double *aShapeCoef = m_pBfmManager->getMutableShapeCoef();
	ceres::CostFunction *costFunction = ShapeCoefReprojErr::create(m_pObservedPoints, m_pBfmManager, m_aLandmarkMap);
	ceres::DynamicAutoDiffCostFunction<ShapeCoefRegTerm> *regTerm = ShapeCoefRegTerm::create(m_pBfmManager);
	regTerm->AddParameterBlock(m_pBfmManager->getNIdPcs());
	regTerm->SetNumResiduals(m_pBfmManager->getNIdPcs());
	problem.AddResidualBlock(costFunction, nullptr, aShapeCoef);
	problem.AddResidualBlock(regTerm, nullptr, aShapeCoef);
	CERES_INIT(N_CERES_ITERATIONS, N_CERES_THREADS, B_CERES_STDCOUT);
	ceres::Solve(options, &problem, &summary);
	BFM_DEBUG("%s\n", summary.BriefReport().c_str());
	m_pBfmManager->genLandmarkBlendshape();
	return (summary.termination_type == ceres::CONVERGENCE);
}


bool MHPEProblem::solveExprCoef() {
	BFM_DEBUG(PRINT_GREEN "#################### Estimate Expression Coefficients ####################\n" COLOR_END);
	ceres::Problem problem;
	double *aExprCoef = m_pBfmManager->getMutableExprCoef();
	ceres::CostFunction *costFunction = ExprCoefReprojErr::create(m_pObservedPoints, m_pBfmManager, m_aLandmarkMap);
	ceres::DynamicAutoDiffCostFunction<ExprCoefRegTerm> *regTerm = ExprCoefRegTerm::create(m_pBfmManager);
	regTerm->AddParameterBlock(m_pBfmManager->getNExprPcs());
	regTerm->SetNumResiduals(m_pBfmManager->getNExprPcs());
	problem.AddResidualBlock(costFunction, nullptr, aExprCoef);
	problem.AddResidualBlock(regTerm, nullptr, aExprCoef);
	CERES_INIT(N_CERES_ITERATIONS, N_CERES_THREADS, B_CERES_STDCOUT);
	ceres::Solve(options, &problem, &summary);
	BFM_DEBUG("%s\n", summary.BriefReport().c_str());
	m_pBfmManager->genLandmarkBlendshape();
	return (summary.termination_type == ceres::CONVERGENCE);
}


bool MHPEProblem::is_close_enough(double *ext_params, double rotation_eps, double translation_eps)
{
	for(int i=0; i<3; i++)
		if(abs(ext_params[i]) > rotation_eps)
			return false;
	for(int i=3; i<6; i++)
		if(abs(ext_params[i]) > translation_eps)
			return false;
	return true;
}


