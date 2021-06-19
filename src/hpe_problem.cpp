#include "hpe_problem.h"

MHPEProblem::MHPEProblem(
	const std::string& sProjectPath,
	const std::string& sBfmH5Path,
	const std::string& sLandmarkIdxPath,
	const std::string& sDlibDetPath
) 
{
	if(this->init(sProjectPath, sBfmH5Path, sLandmarkIdxPath, sDlibDetPath) == Status_Error)
		LOG(ERROR) << "Multi-view head pose estimation problem structure init failed.";
}

Status MHPEProblem::init(
	const std::string& sProjectPath,
	const std::string& sBfmH5Path,
	const std::string& sLandmarkIdxPath,
	const std::string& sDlibDetPath
) 
{
	LOG(INFO) << "Init problem structure of multiple head pose estimation.";

	m_pDataManager = std::make_shared<DataManager>(sProjectPath, sDlibDetPath);
	if(m_pDataManager->getNViews() == 0)
	{
		LOG(ERROR) << "Load data manager failed.";
		return Status_Error;
	}

	std::array<double, N_INT_PARAMS> aIntParams = { };
	aIntParams[0] = aIntParams[1] = m_pDataManager->getF() * ZOOM_SCALE;
	aIntParams[2] = (m_pDataManager->getWidth() * 0.5 + m_pDataManager->getCx()) * ZOOM_SCALE;
	aIntParams[3] = (m_pDataManager->getHeight() * 0.5 + m_pDataManager->getCy()) * ZOOM_SCALE;
	
	m_pBfmManager = std::make_shared<BfmManager>(sBfmH5Path, aIntParams, sLandmarkIdxPath);
	if(m_pBfmManager->getStrModelPath() == "")
	{
		LOG(ERROR) << "Load BFM failed.";
		return Status_Error;
	}
	m_pBfmManager->check();

	return Status_Ok;
}


void MHPEProblem::solve(SolveExtParamsMode mode, double dShapeWeight, double dExprWeight)
{
	LOG(INFO) << "Begin solving multi-view head pose estimation.";
	auto start = std::chrono::system_clock::now();    // Start of solving 

	const auto& nViews = m_pDataManager->getNViews();

	const auto& aRotTypes = m_pDataManager->getRotTypes();
	const auto& a2dImgs = m_pDataManager->getArr2dImgs();
	const auto& a2dTransImgs = m_pDataManager->getArr2dTransImgs();
	const auto& aDets = m_pDataManager->getDets();
	const auto& aTransDets = m_pDataManager->getTransDets();

	std::string sPickedIdx;
	std::vector<DetPair> aDetPairs;
	std::vector<double> vInitPts;
	std::vector<dlib::image_window> winOrigin(nViews), winTrans(nViews);
	bool isConvergence;

	this->rmOutliers();

	LOG(INFO) << "Combine available photos and estimate.";

	tiny_progress::ProgressBar pb(nViews);
	pb.begin(std::ref(std::cout), "Creating windows.");
	for(auto iView = 0; iView < nViews; ++iView)
	{
		pb.update(1, "Create window No. " + std::to_string(iView));		
		this->initWin(
			winOrigin[iView], 
			"Origin-" + std::to_string(iView),
			a2dImgs[iView],
			aDets[iView],
			aRotTypes[iView] != RotateType_Invalid
		);
		this->initWin(
			winTrans[iView],
			"Trans-" + std::to_string(iView),
			a2dTransImgs[iView],
			aTransDets[iView],
			aRotTypes[iView] != RotateType_Invalid
		);

		if(aRotTypes[iView] != RotateType_Invalid)
			aDetPairs.push_back(DetPair(aDets[iView], iView));
	}
	pb.end(std::ref(std::cout), "Creation of windows done");

	sPickedIdx = "";
	for(const auto& detPair : aDetPairs)
		sPickedIdx += " " + std::to_string(detPair.second);
	LOG(INFO) << "Final picked photos include:" << sPickedIdx;

	vInitPts = this->estInit3dPts(aDetPairs);
	double dScMean = this->estInitSc(vInitPts);
	this->estInitExtParams(vInitPts);

	Summary sum;
	do {
		isConvergence = true;
		sum = this->estExtParams(aDetPairs, dScMean);
		isConvergence &= (sum.initial_cost - sum.final_cost < 1e2 * aDetPairs.size());
		sum = this->estShapeCoef(aDetPairs, dShapeWeight);
		isConvergence &= (sum.initial_cost - sum.final_cost < 1e2 * aDetPairs.size());
		sum = this->estExprCoef(aDetPairs, dExprWeight);
		isConvergence &= (sum.initial_cost - sum.final_cost < 1e2 * aDetPairs.size());
	} while(!isConvergence);
	
	// this->showRes(winOrigin);

	// BFM_DEBUG(PRINT_YELLOW "\n[Step 4] Show results.\n" COLOR_END);
	float pfExtParams[N_EXT_PARAMS];
	for(auto i = 0u; i < N_EXT_PARAMS; ++i) pfExtParams[i] = (float)m_pBfmManager->getMutableExtParams()[i];
	Eigen::VectorXf vecPts = m_pBfmManager->getLandmarkCurrentBlendshape().template cast<float>() * (float)m_pBfmManager->getMutableScale();
	
	double totalLoss = 0, loss;
	for(auto i = 0; i < nViews; i++)
	{	
		loss = 0;
		if(aRotTypes[i] == RotateType_Invalid) continue;
		Eigen::VectorXf vecTranPts0 = bfm_utils::TransPoints(pfExtParams, vecPts);
		Eigen::VectorXf vecTranPts1 = bfm_utils::TransPoints(m_pDataManager->getCameraMatrices()[i], vecTranPts0);
		std::vector<dlib::point> aPoints;
		for(unsigned int iLandmark = 0; iLandmark < N_LANDMARKS; iLandmark++) 
		{
			int u = int(m_pBfmManager->getFx() * vecTranPts1(iLandmark * 3) / vecTranPts1(iLandmark * 3 + 2) + m_pBfmManager->getCx());
			int v = int(m_pBfmManager->getFy() * vecTranPts1(iLandmark * 3 + 1) / vecTranPts1(iLandmark * 3 + 2) + m_pBfmManager->getCy());
			aPoints.push_back(dlib::point(u, v));
			loss += std::pow(aDets[i].part(iLandmark).x() - u, 2) + std::pow(aDets[i].part(iLandmark).y() - v, 2);
		}
		LOG(INFO) << "View:\t" << i << "\tMean landmark loss:\t" << std::sqrt(loss / N_DLIB_LANDMARKS); 
		totalLoss += loss;
		winOrigin[i].add_overlay(render_face_detections(dlib::full_object_detection(dlib::rectangle(), aPoints), dlib::rgb_pixel(0, 0, 255)));
	}
	totalLoss /= aDetPairs.size();
	LOG(INFO) << "Mean view loss:\t" << totalLoss << "\tTotal mean landmark loss:\t" << std::sqrt(totalLoss / N_DLIB_LANDMARKS);

	m_pBfmManager->genFace();
	m_pBfmManager->writePly("face_ext_shape.ply", ModelWriteMode_CameraCoord);
	// m_pBfmManager->writePly("avg.ply");

    auto end = std::chrono::system_clock::now();    // End of solving
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    LOG(INFO) << "Cost of solution: "
        << (double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den)
        << " seconds." << std::endl;            

	std::cin.get();
}


void MHPEProblem::estInitExtParams(vector<double>& vPts)
{
	LOG(INFO) << "Estimate Multi-Faces Extrinsic Parameters (Initial)";

	ceres::Problem problem;	
	ceres::CostFunction* costFunction = ExtParams3D3DReprojErr::create(vPts, m_pBfmManager, m_pDataManager);
	ceres::Solver::Options options;
	ceres::Solver::Summary summary;
	auto& aExtParams = m_pBfmManager->getMutableExtParams();

	mhpe::Utils::InitCeresProblem(std::ref(options));	
	problem.AddResidualBlock(costFunction, nullptr, aExtParams.data());
	ceres::Solve(options, &problem, &summary);

	LOG(INFO) << summary.BriefReport();
	
	// Sync result
	m_pBfmManager->genTransMat();	
}


Summary MHPEProblem::estExtParams(const DetPairVector& aObjDets, double scMean)
{
	LOG(INFO) << "Estimate Multi-Faces Extrinsic Parameters" << std::endl;

	double& dScale = this->m_pBfmManager->getMutableScale();
	ceres::Problem problem;	
	ceres::CostFunction *costFunction;
	ceres::Solver::Options options;
	ceres::Solver::Summary summary;

	mhpe::Utils::InitCeresProblem(std::ref(options));
	std::array<double, N_EXT_PARAMS>& aExtParams = this->m_pBfmManager->getMutableExtParams();
	
	costFunction = MultiExtParamsReprojErr::create(aObjDets, m_pBfmManager.get(), m_pDataManager.get(), scMean);
	problem.AddResidualBlock(costFunction, nullptr, aExtParams.data(), &dScale);
	ceres::Solve(options, &problem, &summary);
	
	BFM_DEBUG("%s\n", summary.BriefReport().c_str());
	LOG(INFO) << "Scale: " << dScale << std::endl;
	
	// Sync result
	m_pBfmManager->genTransMat();
	
	return summary;
}

Summary MHPEProblem::estShapeCoef(const DetPairVector& aObjDets, double dShapeWeight) 
{
	LOG(INFO) << "Estimate Multi-Faces Shape Coefficients" << std::endl;

	double *pShapeCoef;
	ceres::Problem problem;	
	ceres::CostFunction *costFunction;
	ceres::Solver::Options options;
	ceres::Solver::Summary summary;

	mhpe::Utils::InitCeresProblem(std::ref(options));

	pShapeCoef = m_pBfmManager->getMutableShapeCoef();
	
	costFunction = MultiShapeCoefReprojErr::create(aObjDets, m_pBfmManager.get(), m_pDataManager.get(), dShapeWeight);
	problem.AddResidualBlock(costFunction, nullptr, pShapeCoef);
	ceres::Solve(options, &problem, &summary);

	LOG(INFO) << summary.BriefReport() << std::endl;
 
	// Sync	result
	m_pBfmManager->genLandmarkBlendshape();

	return summary;
}


Summary MHPEProblem::estExprCoef(const DetPairVector& aObjDets, double dExprWeight) 
{
	LOG(INFO) << "Estimate Multi-Faces Expression Coefficients" << std::endl;

	double *pExprCoef;
	ceres::Problem problem;	
	ceres::CostFunction *costFunction;
	ceres::Solver::Options options;
	ceres::Solver::Summary summary;

	mhpe::Utils::InitCeresProblem(std::ref(options));
	
	pExprCoef = m_pBfmManager->getMutableExprCoef();

	costFunction = MultiExprCoefReprojErr::create(aObjDets, m_pBfmManager.get(), m_pDataManager.get(), dExprWeight);
	problem.AddResidualBlock(costFunction, nullptr, pExprCoef);
	ceres::Solve(options, &problem, &summary);

	LOG(INFO) << summary.BriefReport() << std::endl;

	// Sync	result
	m_pBfmManager->genLandmarkBlendshape();

	return summary;
}


// bool MHPEProblem::solveExtParams(long mode, double ca, double cb) 
// {
// 	BFM_DEBUG(PRINT_GREEN "#################### Estimate Extrinsic Parameters ####################\n" COLOR_END);
// 	if(mode & SolveExtParamsMode_UseOpenCV)
// 	{
// 		const double *aIntParams = m_pBfmManager->getIntParams().data();
// 		double aCameraMatrix[3][3] = {
// 			{aIntParams[0], 0.0, aIntParams[2]},
// 			{0.0, aIntParams[1], aIntParams[3]},
// 			{0.0, 0.0, 1.0}
// 		};
// 		cv::Mat matCameraMatrix = cv::Mat(3, 3, CV_64FC1, aCameraMatrix);		
// 		#ifdef _DEBUG
// 		std::cout << "camera matrix: " << matCameraMatrix << std::endl;
// 		#endif
// 		std::vector<float> aDistCoef(0);
// 		std::vector<cv::Point3f> out;
// 		std::vector<cv::Point2f> in;
// 		cv::Mat rvec, tvec;
// 		const VectorXd& vecLandmarkBlendshape = m_pBfmManager->getLandmarkCurrentBlendshape();
// 		for(unsigned int iLandmark = 0; iLandmark < 68; iLandmark++) {
// 			out.push_back(cv::Point3f(
// 				vecLandmarkBlendshape(iLandmark * 3), 
// 				vecLandmarkBlendshape(iLandmark * 3 + 1), 
// 				vecLandmarkBlendshape(iLandmark * 3 + 2)));
// 			in.push_back(cv::Point2f(m_pObservedPoints->part(iLandmark).x(), m_pObservedPoints->part(iLandmark).y()));
// 		}
// 		cv::solvePnP(out, in, matCameraMatrix, aDistCoef, rvec, tvec);
// 		cv::Rodrigues(rvec, rvec);
// 		#ifdef _DEBUG
// 		std::cout << rvec << std::endl;
// 		std::cout << tvec << std::endl;
// 		#endif
// 		m_pBfmManager->setMatR(rvec);
// 		m_pBfmManager->setVecT(tvec);
// 		m_pBfmManager->genExtParams();
// 		return true;
// 	}
// 	else if(mode & SolveExtParamsMode_UseLinearizedRadians)
// 	{
// 		BFM_DEBUG("solve -> external parameters (linealized)\n");
// 		if(mode & SolveExtParamsMode_UseDlt)
// 		{
// 			BFM_DEBUG("	1) estimate initial values by using DLT algorithm.\n");
// 			// dlt();
// 		}
// 		else
// 		{
// 			BFM_DEBUG("	1) initial values have been set in advance or are 0s.\n");
// 		}
		
// 		m_pBfmManager->genTransMat();	

// 		CERES_INIT(N_CERES_ITERATIONS, N_CERES_THREADS, B_CERES_STDCOUT);
// 		while(true) 
// 		{
// 			ceres::Problem problem;
// 			double aSmallExtParams[6] = { 0.f };
// 			ceres::CostFunction *costFunction = LinearizedExtParamsReprojErr::create(m_pObservedPoints, m_pBfmManager.get(), m_aLandmarkMap, ca, cb);
// 			problem.AddResidualBlock(costFunction, nullptr, aSmallExtParams);
// 			ceres::Solve(options, &problem, &summary);
// 			BFM_DEBUG("%s\n", summary.BriefReport().c_str());
			
// 			if(is_close_enough(aSmallExtParams, 0, 0)) 
// 			{
// 				#ifdef _DEBUG
// 				bfm_utils::PrintArr(aSmallExtParams, 6);
// 				std::cout << summary.BriefReport() << std::endl;
// 				#endif
// 				break; 
// 			}

// 			m_pBfmManager->accExtParams(aSmallExtParams);
			
// 			#ifdef _DEBUG
// 			bfm_utils::PrintArr(aSmallExtParams, 6);							
// 			#endif
// 		}
// 		m_pBfmManager->genExtParams();
// 		return (summary.termination_type == ceres::CONVERGENCE);
// 	}
// 	else
// 	{
// 		#ifndef HPE_SHUT_UP
// 		std::cout << "solve -> external parameters" << std::endl;	
// 		std::cout << "init ceres solve - ";
// 		#endif

// 		#ifndef HPE_SHUT_UP
// 		if(mode & SolveExtParamsMode_UseDlt)
// 		{
// 			std::cout << "	1) esitimate initial values by using DLT algorithm." << std::endl;
// 			// dlt();
// 		}
// 		else
// 		{
// 			std::cout << "	1) initial values have been set in advance or are 0s." << std::endl;
// 		}
// 		#else
// 		if(mode & SolveExtParamsMode_UseDlt) dlt();
// 		#endif

// 		ceres::Problem problem;
// 		double *ext_params = m_pBfmManager->getMutableExtParams().data();
// 		ceres::CostFunction *costFunction = ExtParamsReprojErr::create(m_pObservedPoints, m_pBfmManager.get(), m_aLandmarkMap);
// 		problem.AddResidualBlock(costFunction, nullptr, ext_params);
// 		ceres::Solver::Options options;
// 		options.max_num_iterations = 100;
// 		options.num_threads = 8;
// 		options.minimizer_progress_to_stdout = true;
// 		ceres::Solver::Summary summary;
// 		#ifndef HPE_SHUT_UP
// 		std::cout << "success" << std::endl;
// 		#endif
// 		ceres::Solve(options, &problem, &summary);
// 		#ifndef HPE_SHUT_UP
// 		std::cout << summary.BriefReport() << std::endl;
// 		#endif
// 		m_pBfmManager->genTransMat();
// 		return (summary.termination_type == ceres::CONVERGENCE);
// 	}
// }


// bool MHPEProblem::solveShapeCoef() {
// 	BFM_DEBUG(PRINT_GREEN "#################### Estimate Shape Coefficients ####################\n" COLOR_END);
// 	ceres::Problem problem;
// 	double *aShapeCoef = m_pBfmManager->getMutableShapeCoef();
// 	ceres::CostFunction *costFunction = ShapeCoefReprojErr::create(m_pObservedPoints, m_pBfmManager.get(), m_aLandmarkMap);
// 	ceres::DynamicAutoDiffCostFunction<ShapeCoefRegTerm> *regTerm = ShapeCoefRegTerm::create(m_pBfmManager.get());
// 	regTerm->AddParameterBlock(m_pBfmManager->getNIdPcs());
// 	regTerm->SetNumResiduals(m_pBfmManager->getNIdPcs());
// 	problem.AddResidualBlock(costFunction, nullptr, aShapeCoef);
// 	problem.AddResidualBlock(regTerm, nullptr, aShapeCoef);
// 	CERES_INIT(N_CERES_ITERATIONS, N_CERES_THREADS, B_CERES_STDCOUT);
// 	ceres::Solve(options, &problem, &summary);
// 	BFM_DEBUG("%s\n", summary.BriefReport().c_str());
// 	m_pBfmManager->genLandmarkBlendshape();
// 	return (summary.termination_type == ceres::CONVERGENCE);
// }


// bool MHPEProblem::solveExprCoef() {
// 	BFM_DEBUG(PRINT_GREEN "#################### Estimate Expression Coefficients ####################\n" COLOR_END);
// 	ceres::Problem problem;
// 	double *aExprCoef = m_pBfmManager->getMutableExprCoef();
// 	ceres::CostFunction *costFunction = ExprCoefReprojErr::create(m_pObservedPoints, m_pBfmManager.get(), m_aLandmarkMap);
// 	ceres::DynamicAutoDiffCostFunction<ExprCoefRegTerm> *regTerm = ExprCoefRegTerm::create(m_pBfmManager.get());
// 	regTerm->AddParameterBlock(m_pBfmManager->getNExprPcs());
// 	regTerm->SetNumResiduals(m_pBfmManager->getNExprPcs());
// 	problem.AddResidualBlock(costFunction, nullptr, aExprCoef);
// 	problem.AddResidualBlock(regTerm, nullptr, aExprCoef);
// 	CERES_INIT(N_CERES_ITERATIONS, N_CERES_THREADS, B_CERES_STDCOUT);
// 	ceres::Solve(options, &problem, &summary);
// 	BFM_DEBUG("%s\n", summary.BriefReport().c_str());
// 	m_pBfmManager->genLandmarkBlendshape();
// 	return (summary.termination_type == ceres::CONVERGENCE);
// }


// bool MHPEProblem::is_close_enough(double *ext_params, double rotation_eps, double translation_eps)
// {
// 	for(int i=0; i<3; i++)
// 		if(abs(ext_params[i]) > rotation_eps)
// 			return false;
// 	for(int i=3; i<6; i++)
// 		if(abs(ext_params[i]) > translation_eps)
// 			return false;
// 	return true;
// }


std::vector<double> MHPEProblem::estInit3dPts(const std::vector<DetPair>& vDetPairs)
{
	LOG(INFO) << "Using 3d-3d to estimate initial 3d point cloud.";
	
	ceres::Problem problem;	
	ceres::CostFunction* costFunction = PointReprojErr::create(vDetPairs, m_pBfmManager, m_pDataManager);
	ceres::Solver::Options options;
	ceres::Solver::Summary summary;	

	std::vector<double> vPts(m_pBfmManager->getNLandmarks() * 3, 0.0);
	mhpe::Utils::InitCeresProblem(std::ref(options));

	problem.AddResidualBlock(costFunction, nullptr, vPts.data());
	ceres::Solve(options, &problem, &summary);
	LOG(INFO) << summary.BriefReport();	

	std::ofstream out("init.off");
	out << "COFF\n";
	out << vPts.size() / 3 << " 0 0\n";
	for(int i = 0; i < vPts.size() / 3; i++)
		out << vPts[i * 3] << " " << vPts[i * 3 + 1] << " " << vPts[i * 3 + 2] << "\n";
	out.close();

	return vPts;
}


double MHPEProblem::estInitSc(const std::vector<double>& vPts)
{
	LOG(INFO) << "Using intuition to estimate initial scale.";

	std::array<int, 2> idxPairHor = {27, 8};
	std::array<int, 2> idxPairVer = {0, 16};
	double x1, y1, z1, x2, y2, z2;
	double dis1, dis2;
	double scHor, scVer;
	double& initSc = m_pBfmManager->getMutableScale();
	const auto& modelPoints = m_pBfmManager->getLandmarkCurrentBlendshape();

	x1 = vPts[idxPairHor[0] * 3];
	y1 = vPts[idxPairHor[0] * 3 + 1];
	z1 = vPts[idxPairHor[0] * 3 + 2];
	x2 = vPts[idxPairHor[1] * 3];
	y2 = vPts[idxPairHor[1] * 3 + 1];
	z2 = vPts[idxPairHor[1] * 3 + 2];
	dis1 = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2);
	x1 = modelPoints(idxPairHor[0] * 3);
	y1 = modelPoints(idxPairHor[0] * 3 + 1);
	z1 = modelPoints(idxPairHor[0] * 3 + 2);
	x2 = modelPoints(idxPairHor[1] * 3);
	y2 = modelPoints(idxPairHor[1] * 3 + 1);
	z2 = modelPoints(idxPairHor[1] * 3 + 2);
	dis2 = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2);
	scHor = sqrt(dis1) / sqrt(dis2);
	x1 = vPts[idxPairVer[0] * 3];
	y1 = vPts[idxPairVer[0] * 3 + 1];
	z1 = vPts[idxPairVer[0] * 3 + 2];
	x2 = vPts[idxPairVer[1] * 3];
	y2 = vPts[idxPairVer[1] * 3 + 1];
	z2 = vPts[idxPairVer[1] * 3 + 2];
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
	return initSc;
}


void MHPEProblem::initWin(
	dlib::image_window& win, 
	const std::string& sTitle,
	const Arr2d& img,
	const ObjDet& objDet, 
	bool bIsValid)
{
	if(!bIsValid)
		win.close_window();
	else
	{
		win.clear_overlay();
		win.set_title(sTitle);
		win.set_image(img);
		win.add_overlay(dlib::render_face_detections(objDet));		
	}
}


void MHPEProblem::rmOutliers()
{
	LOG(INFO) << "Remove outliers.";

	auto& aRotTypes = m_pDataManager->getRotTypes();
	auto nViews= m_pDataManager->getNViews();
	for(auto iView = 0; iView < nViews; ++iView)
	{
		if(iView == 0
			|| iView == 1
			|| iView == 2
			|| iView == 3
			|| iView == 9
			|| iView == 7 // 倒着的脸也能检测出特征点
			|| iView == 11
			|| iView == 13
			|| iView == 19
			|| iView == 20
			|| iView == 22
			|| iView == 23
			|| iView == 4 // tmp
			|| iView == 21 // tmp
			|| iView == 18 // tmp
		)
			aRotTypes[iView] = RotateType_Invalid;
	}
}

void MHPEProblem::showRes(std::vector<dlib::image_window>& vWins)
{
	const auto& nViews = m_pDataManager->getNViews();
	auto& aRotTypes = m_pDataManager->getRotTypes();
	const auto& aMatCams = m_pDataManager->getCameraMatrices();
	Eigen::Matrix<float, Dynamic, 1> vPts = m_pBfmManager->getLandmarkCurrentBlendshapeTransformed()
			  							  .template cast<float>()
			  							  * static_cast<float>(m_pBfmManager->getMutableScale());
	auto cx = m_pBfmManager->getCx();
	auto cy = m_pBfmManager->getCy();
	auto fx = m_pBfmManager->getFx();
	auto fy = m_pBfmManager->getFy();

	for(auto iView = 0; iView < nViews; ++iView)
	{
		if(aRotTypes[iView] == RotateType_Invalid)
			continue;
		Eigen::VectorXf vTranPts = bfm_utils::TransPoints(aMatCams[iView], vPts);
		std::vector<dlib::point> pts;
		for(auto j = 0u; j < N_LANDMARKS; j++) 
		{
			auto invZ = 1.0 / vTranPts(j * 3 + 2);
			auto u = fx * vTranPts(j * 3) * invZ + cx;
			auto v = fy * vTranPts(j * 3 + 1) * invZ + cy;
			pts.emplace_back(u, v);
		}
		vWins[iView].add_overlay(dlib::render_face_detections(ObjDet(dlib::rectangle(), pts), dlib::rgb_pixel(0, 0, 255)));		
	}
}