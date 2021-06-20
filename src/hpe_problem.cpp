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

	const auto nViews = m_pDataManager->getNViews();
	const auto nLandmarks = m_pBfmManager->getNLandmarks();

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

	std::vector<std::vector<bool>> validList(nViews, std::vector<bool>(nLandmarks, true));

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
	std::size_t i = 0;
	while(true)
	{
		++i;
		isConvergence = true;
		sum = this->estShapeCoef(aDetPairs, dShapeWeight, validList);
		isConvergence &= (sum.initial_cost - sum.final_cost < 1e2);
		sum = this->estExprCoef(aDetPairs, dExprWeight, validList);
		isConvergence &= (sum.initial_cost - sum.final_cost < 1e2);
		sum = this->estExtParams(aDetPairs, dScMean, validList);
		isConvergence &= (sum.initial_cost - sum.final_cost < 1e2);

		if(isConvergence || i == MAX_N_ITERATIONS) break;

		rm2dLandmarkOutliers(validList);
	} 
	
	this->showRes(winOrigin, validList, aDetPairs);

	m_pBfmManager->genFace();
	m_pBfmManager->writePly("output.ply", ModelWriteMode_CameraCoord);

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


Summary MHPEProblem::estExtParams(const DetPairVector& aObjDets, double scMean, std::vector<std::vector<bool>>& validList)
{
	LOG(INFO) << "Estimate Multi-Faces Extrinsic Parameters" << std::endl;

	double& dScale = this->m_pBfmManager->getMutableScale();
	ceres::Problem problem;	
	ceres::CostFunction *costFunction;
	ceres::Solver::Options options;
	ceres::Solver::Summary summary;

	mhpe::Utils::InitCeresProblem(std::ref(options));
	std::array<double, N_EXT_PARAMS>& aExtParams = this->m_pBfmManager->getMutableExtParams();
	
	costFunction = MultiExtParamsReprojErr::create(aObjDets, m_pBfmManager.get(), m_pDataManager.get(), scMean, validList);
	problem.AddResidualBlock(costFunction, nullptr, aExtParams.data(), &dScale);
	ceres::Solve(options, &problem, &summary);
	
	BFM_DEBUG("%s\n", summary.BriefReport().c_str());
	LOG(INFO) << "Scale: " << dScale << std::endl;
	
	// Sync result
	m_pBfmManager->genTransMat();
	
	return summary;
}

Summary MHPEProblem::estShapeCoef(const DetPairVector& aObjDets, double dShapeWeight, std::vector<std::vector<bool>>& validList) 
{
	LOG(INFO) << "Estimate Multi-Faces Shape Coefficients" << std::endl;

	double *pShapeCoef;
	ceres::Problem problem;	
	ceres::CostFunction *costFunction;
	ceres::Solver::Options options;
	ceres::Solver::Summary summary;

	mhpe::Utils::InitCeresProblem(std::ref(options));

	pShapeCoef = m_pBfmManager->getMutableShapeCoef();
	
	costFunction = MultiShapeCoefReprojErr::create(aObjDets, m_pBfmManager.get(), m_pDataManager.get(), dShapeWeight, validList);
	problem.AddResidualBlock(costFunction, nullptr, pShapeCoef);
	ceres::Solve(options, &problem, &summary);

	LOG(INFO) << summary.BriefReport() << std::endl;
 
	// Sync	result
	m_pBfmManager->genLandmarkBlendshape();

	return summary;
}


Summary MHPEProblem::estExprCoef(const DetPairVector& aObjDets, double dExprWeight, std::vector<std::vector<bool>>& validList) 
{
	LOG(INFO) << "Estimate Multi-Faces Expression Coefficients" << std::endl;

	double *pExprCoef;
	ceres::Problem problem;	
	ceres::CostFunction *costFunction;
	ceres::Solver::Options options;
	ceres::Solver::Summary summary;

	mhpe::Utils::InitCeresProblem(std::ref(options));
	
	pExprCoef = m_pBfmManager->getMutableExprCoef();

	costFunction = MultiExprCoefReprojErr::create(aObjDets, m_pBfmManager.get(), m_pDataManager.get(), dExprWeight, validList);
	problem.AddResidualBlock(costFunction, nullptr, pExprCoef);
	ceres::Solve(options, &problem, &summary);

	LOG(INFO) << summary.BriefReport() << std::endl;

	// Sync	result
	m_pBfmManager->genLandmarkBlendshape();

	return summary;
}


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

void MHPEProblem::showRes(
	std::vector<dlib::image_window>& vWins,
	const std::vector<std::vector<bool>>& validList,
	const std::vector<DetPair>& aDetPairs) 
{
	const auto nViews = m_pDataManager->getNViews();
	const auto& aDets = m_pDataManager->getDets();
	const auto& aRotTypes = m_pDataManager->getRotTypes();

	float pfExtParams[N_EXT_PARAMS];
	for(auto i = 0u; i < N_EXT_PARAMS; ++i) pfExtParams[i] = (float)m_pBfmManager->getMutableExtParams()[i];
	Eigen::VectorXf vecPts = m_pBfmManager->getLandmarkCurrentBlendshape().template cast<float>() * (float)m_pBfmManager->getMutableScale();
	
	std::size_t nValidLandmarks;
	double totalLoss = 0, loss;
	for(auto i = 0; i < nViews; i++)
	{	
		loss = 0;
		nValidLandmarks = 0;

		if(aRotTypes[i] == RotateType_Invalid) continue;
		Eigen::VectorXf vecTranPts0 = bfm_utils::TransPoints(pfExtParams, vecPts);
		Eigen::VectorXf vecTranPts1 = bfm_utils::TransPoints(m_pDataManager->getCameraMatrices()[i], vecTranPts0);
		std::vector<dlib::point> aSrcPts, aDstPts, aPts;
		for(unsigned int iLandmark = 0; iLandmark < N_LANDMARKS; iLandmark++) 
		{			
			++nValidLandmarks;
			int u = int(m_pBfmManager->getFx() * vecTranPts1(iLandmark * 3) / vecTranPts1(iLandmark * 3 + 2) + m_pBfmManager->getCx());
			int v = int(m_pBfmManager->getFy() * vecTranPts1(iLandmark * 3 + 1) / vecTranPts1(iLandmark * 3 + 2) + m_pBfmManager->getCy());
			aPts.emplace_back(u, v);
			if(!validList[i][iLandmark]) continue;
			aSrcPts.push_back(dlib::point(u, v));
			aDstPts.emplace_back(aDets[i].part(iLandmark).x(), aDets[i].part(iLandmark).y());
			loss += std::pow(aDets[i].part(iLandmark).x() - u, 2) + std::pow(aDets[i].part(iLandmark).y() - v, 2);
		}
		loss /= nValidLandmarks;
		LOG(INFO) << "View:\t" << i << "\tMean landmark loss:\t" << std::sqrt(loss); 
		totalLoss += loss;
		vWins[i].add_overlay(dlib::render_face_detections(ObjDet(dlib::rectangle(), aPts), dlib::rgb_pixel(255, 0, 255)));
		vWins[i].add_overlay(utils::renderPts(aDstPts, 2.0, dlib::rgb_pixel(255, 255, 0)));
		vWins[i].add_overlay(utils::renderPts(aSrcPts, 2.0, dlib::rgb_pixel(0, 0, 255)));
	}
	totalLoss /= aDetPairs.size();
	LOG(INFO) << "Total mean landmark loss:\t" << std::sqrt(totalLoss);
}


void MHPEProblem::rm2dLandmarkOutliers(std::vector<std::vector<bool>>& validList)
{
	const auto nViews = m_pDataManager->getNViews();
	const auto& aDets = m_pDataManager->getDets();
	const auto& aRotTypes = m_pDataManager->getRotTypes();

	float pfExtParams[N_EXT_PARAMS];
	for(auto i = 0u; i < N_EXT_PARAMS; ++i) pfExtParams[i] = (float)m_pBfmManager->getMutableExtParams()[i];
	Eigen::VectorXf vecPts = m_pBfmManager->getLandmarkCurrentBlendshape().template cast<float>() * (float)m_pBfmManager->getMutableScale();
	std::string sOutliers;

	static double threshold = 8.0;

	for(auto i = 0; i < nViews; i++)
	{	
		if(aRotTypes[i] == RotateType_Invalid) continue;
		sOutliers = std::to_string(i) + ": ";
		Eigen::VectorXf vecTranPts0 = bfm_utils::TransPoints(pfExtParams, vecPts);
		Eigen::VectorXf vecTranPts1 = bfm_utils::TransPoints(m_pDataManager->getCameraMatrices()[i], vecTranPts0);
		std::vector<dlib::point> aPoints;
		for(unsigned int iLandmark = 0; iLandmark < N_LANDMARKS; iLandmark++) 
		{
			if(!validList[i][iLandmark])
				continue;
			int u = int(m_pBfmManager->getFx() * vecTranPts1(iLandmark * 3) / vecTranPts1(iLandmark * 3 + 2) + m_pBfmManager->getCx());
			int v = int(m_pBfmManager->getFy() * vecTranPts1(iLandmark * 3 + 1) / vecTranPts1(iLandmark * 3 + 2) + m_pBfmManager->getCy());
			aPoints.push_back(dlib::point(u, v));
			auto loss = std::sqrt(std::pow(aDets[i].part(iLandmark).x() - u, 2) + std::pow(aDets[i].part(iLandmark).y() - v, 2));
			
			if(loss > threshold)
			{
				validList[i][iLandmark] = false;
				sOutliers += std::to_string(iLandmark) + ", ";
			} 
		}
		if(sOutliers.size() > 4)
		{
			LOG(INFO) << "Remove following outliers:";
			LOG(INFO) << "\t" << sOutliers; 
		}
	}
	threshold /= 2.0;
}