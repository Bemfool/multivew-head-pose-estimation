#ifndef HPE_EXT_PARAMS_REPROJ_ERR_H
#define HPE_EXT_PARAMS_REPROJ_ERR_H

#include "bfm_manager.h"
#include "data_manager.h"
#include "ceres/ceres.h"
#include "db_params.h"
#include "io_utils.h"

using FullObjectDetection = dlib::full_object_detection;
using Eigen::VectorXd;
using Eigen::Matrix;
using Eigen::Dynamic;

class MultiExtParamsReprojErr 
{
public:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
	MultiExtParamsReprojErr(
		const std::vector<std::pair<FullObjectDetection, int>> &aObjDetections, 
		BfmManager *model, 
		DataManager* pDataManager,
		double scMean,
		std::vector<std::vector<bool>>& validList) : 
		m_aObjDetections(aObjDetections), 
	    m_pModel(model), 
		m_pDataManager(pDataManager),
		m_scMean(scMean),
		m_validList(validList) { }
	

    template<typename _Tp>
	bool operator () (const _Tp* const pExtParams, const _Tp* const pScale, _Tp* aResiduals) const 
	{
		_Tp fx, fy, cx, cy;
		_Tp u, v;
		_Tp invZ;
		int iFace;
		
		// Fetch intrinsic parameters
		fx = static_cast<_Tp>(m_pModel->getFx());
		fy = static_cast<_Tp>(m_pModel->getFy());
		cx = static_cast<_Tp>(m_pModel->getCx());
		cy = static_cast<_Tp>(m_pModel->getCy());

		// Fetch camera matrix array
		const std::vector<Eigen::Matrix4f>& aMatCams = m_pDataManager->getCameraMatrices();

		// Transform
		const _Tp& scale = *pScale;
		const Matrix<_Tp, Dynamic, 1> vecPts = m_pModel->getLandmarkCurrentBlendshape().template cast<_Tp>() * scale;
		const Matrix<_Tp, Dynamic, 1> vecRotPts = bfm_utils::TransPoints(pExtParams, vecPts);

		// Compute Euler Distance between landmark and transformed model points in every photos
		for(auto i = 0u; i < m_aObjDetections.size(); i++)
		{
			iFace = m_aObjDetections[i].second;
 			const Eigen::Matrix<_Tp, 4, 4> matCam = aMatCams[iFace].template cast<_Tp>();
			const Matrix<_Tp, Dynamic, 1> vecTranPts = bfm_utils::TransPoints(matCam, vecRotPts);

			for(auto j = 0; j < N_LANDMARKS; j++) 
			{
				if(m_validList[iFace][j])
				{
					invZ = static_cast<_Tp>(1.0) / vecTranPts(j * 3 + 2);
					u = fx * vecTranPts(j * 3) * invZ + cx;
					v = fy * vecTranPts(j * 3 + 1) * invZ + cy;
					aResiduals[i * N_LANDMARKS * 2 + j * 2] = static_cast<_Tp>(m_aObjDetections[i].first.part(j).x()) - u;
					aResiduals[i * N_LANDMARKS * 2 + j * 2 + 1] = static_cast<_Tp>(m_aObjDetections[i].first.part(j).y()) - v;
				}
				else
				{
					aResiduals[i * N_LANDMARKS * 2 + j * 2] = aResiduals[i * N_LANDMARKS * 2 + j * 2 + 1] = static_cast<_Tp>(0.0);					
				}
			}
		}

		// Empty residuals
		for(auto i = m_aObjDetections.size(); i < N_PHOTOS; i++)
			for(auto j = 0u; j < N_LANDMARKS; j++)
				aResiduals[i * N_LANDMARKS * 2 + j * 2]
				 = aResiduals[i * N_LANDMARKS * 2 + j * 2 + 1]
				 = static_cast<_Tp>(0.0);

		// Regularization
		aResiduals[N_RES] = static_cast<_Tp>(m_scWeight) * (scale - static_cast<_Tp>(m_scMean));

		return true;
	}

	static ceres::CostFunction *create(
		const std::vector<std::pair<dlib::full_object_detection, int>> &aObjDetections, 
		BfmManager *model, 
		DataManager* pDataManager,
		double scMean,
		std::vector<std::vector<bool>>& validList) 
	{
		return (new ceres::AutoDiffCostFunction<MultiExtParamsReprojErr, N_LANDMARKS * 2 * N_PHOTOS + 1, N_EXT_PARAMS, 1>(
			new MultiExtParamsReprojErr(aObjDetections, model, pDataManager, scMean, validList)));
	}

private:
	const std::vector<std::pair<dlib::full_object_detection, int>>& m_aObjDetections;
    BfmManager *m_pModel;
	DataManager* m_pDataManager;
	double m_scWeight = 1e6;
	double m_scMean = 0.0;
	std::vector<std::vector<bool>>& m_validList;
};


class ExtParams3D3DReprojErr 
{
public:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
	ExtParams3D3DReprojErr(
		std::vector<double>& vPts,  
		std::shared_ptr<BfmManager>& pBfmManager, 
		std::shared_ptr<DataManager>& pDataManager) : 
		m_vPts(vPts),
	    m_pBfmManager(pBfmManager), 
		m_pDataManager(pDataManager) { }
	
    template<typename _Tp>
	bool operator () (const _Tp* const pExtParams, _Tp* aResiduals) const 
	{
		_Tp fx, fy, cx, cy, sc;
		_Tp u, v;
		_Tp invZ;
		int iFace;
		
		// Fetch intrinsic parameters
		fx = static_cast<_Tp>(m_pBfmManager->getFx());
		fy = static_cast<_Tp>(m_pBfmManager->getFy());
		cx = static_cast<_Tp>(m_pBfmManager->getCx());
		cy = static_cast<_Tp>(m_pBfmManager->getCy());

		// Fetch camera matrix array
		const auto& aMatCams = m_pDataManager->getCameraMatrices();
		sc = static_cast<_Tp>(m_pBfmManager->getScale());

		// Transform
		const Matrix<_Tp, Dynamic, 1> vPts = m_pBfmManager->getLandmarkCurrentBlendshape().template cast<_Tp>() * sc;
		const Matrix<_Tp, Dynamic, 1> vRotPts = bfm_utils::TransPoints(pExtParams, vPts);

		for(auto i = 0u; i < N_LANDMARKS; i++)
		{
			aResiduals[i * 3] = vRotPts(i * 3) - m_vPts[i * 3];
			aResiduals[i * 3 + 1] = vRotPts(i * 3 + 1) - m_vPts[i * 3 + 1];
			aResiduals[i * 3 + 2] = vRotPts(i * 3 + 2) - m_vPts[i * 3 + 2];
		}

		return true;
	}

	static ceres::CostFunction *create(
		std::vector<double>& vPts, 
		std::shared_ptr<BfmManager>& pBfmManager, 
		std::shared_ptr<DataManager>& pDataManager) 
	{
		return (new ceres::AutoDiffCostFunction<ExtParams3D3DReprojErr, N_LANDMARKS * 3, N_EXT_PARAMS>(
			new ExtParams3D3DReprojErr(vPts, pBfmManager, pDataManager)));
	}

private:
	std::vector<double> m_vPts;
    std::shared_ptr<BfmManager> m_pBfmManager;
	std::shared_ptr<DataManager> m_pDataManager;
	double m_scWeight = 1e6;
};



class ExtParamsReprojErr 
{
public:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
	ExtParamsReprojErr(dlib::full_object_detection *observedPoints, BfmManager *model, std::vector<unsigned int> &aLandmarkMap) 
	: m_pObservedPoints(observedPoints), m_pModel(model), m_aLandmarkMap(aLandmarkMap) { }
	
    template<typename _Tp>
	bool operator () (const _Tp* const aExtParams, _Tp* aResiduals) const 
	{
		_Tp fx = _Tp(m_pModel->getFx()), fy = _Tp(m_pModel->getFy());
		_Tp cx = _Tp(m_pModel->getCx()), cy = _Tp(m_pModel->getCy());
		const VectorXd vecLandmarkBlendshape = m_pModel->getLandmarkCurrentBlendshape();
		const Matrix<_Tp, Dynamic, 1> vecLandmarkBlendshapeTransformed = bfm_utils::TransPoints(aExtParams, vecLandmarkBlendshape);

		for(int iLandmark = 0; iLandmark < N_LANDMARKS; iLandmark++) 
		{
			int iDlibLandmarkIdx = m_aLandmarkMap[iLandmark];
			_Tp u = fx * vecLandmarkBlendshapeTransformed(iLandmark * 3) / vecLandmarkBlendshapeTransformed(iLandmark * 3 + 2) + cx;
			_Tp v = fy * vecLandmarkBlendshapeTransformed(iLandmark * 3 + 1) / vecLandmarkBlendshapeTransformed(iLandmark * 3 + 2) + cy;
			aResiduals[iLandmark * 2] = _Tp(m_pObservedPoints->part(iDlibLandmarkIdx).x()) - u;
			aResiduals[iLandmark * 2 + 1] = _Tp(m_pObservedPoints->part(iDlibLandmarkIdx).y()) - v;
		}
		return true;
	}

	static ceres::CostFunction *create(dlib::full_object_detection *observedPoints, BfmManager *model, std::vector<unsigned int> aLandmarkMap) 
	{
		return (new ceres::AutoDiffCostFunction<ExtParamsReprojErr, N_LANDMARKS * 2, N_EXT_PARAMS>(
			new ExtParamsReprojErr(observedPoints, model, aLandmarkMap)));
	}

private:
	dlib::full_object_detection *m_pObservedPoints;
    BfmManager *m_pModel;
	std::vector<unsigned int> m_aLandmarkMap;
};


class LinearizedExtParamsReprojErr 
{
public:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
	LinearizedExtParamsReprojErr(dlib::full_object_detection *observedPoints, BfmManager *model, std::vector<unsigned int> aLandmarkMap) 
	: m_pObservedPoints(observedPoints), m_pModel(model), m_aLandmarkMap(aLandmarkMap) { }

	LinearizedExtParamsReprojErr(dlib::full_object_detection *observedPoints, BfmManager *model, std::vector<unsigned int> aLandmarkMap, double a, double b) 
	: m_pObservedPoints(observedPoints), m_pModel(model), m_aLandmarkMap(aLandmarkMap), m_dRotWeight(a), m_dTranWeight(b) {}


    template<typename _Tp>
	bool operator () (const _Tp* const aExtParams, _Tp* aResiduals) const 
	{
		_Tp fx = _Tp(m_pModel->getFx()), fy = _Tp(m_pModel->getFy());
		_Tp cx = _Tp(m_pModel->getCx()), cy = _Tp(m_pModel->getCy());

		const VectorXd vecLandmarkBlendshape = m_pModel->getLandmarkCurrentBlendshapeTransformed();
		const Matrix<_Tp, Dynamic, 1> vecLandmarkBlendshapeTransformed = bfm_utils::TransPoints(aExtParams, vecLandmarkBlendshape, true);

		for(int iLandmark = 0; iLandmark < N_LANDMARKS; iLandmark++) 
		{
			int iDlibLandmarkIdx = m_aLandmarkMap[iLandmark];
			_Tp u = fx * vecLandmarkBlendshapeTransformed(iLandmark * 3) / vecLandmarkBlendshapeTransformed(iLandmark * 3 + 2) + cx;
			_Tp v = fy * vecLandmarkBlendshapeTransformed(iLandmark * 3 + 1) / vecLandmarkBlendshapeTransformed(iLandmark * 3 + 2) + cy;
			aResiduals[iLandmark * 2] = _Tp(m_pObservedPoints->part(iDlibLandmarkIdx).x()) - u;
			aResiduals[iLandmark * 2 + 1] = _Tp(m_pObservedPoints->part(iDlibLandmarkIdx).y()) - v;
		}

		/* regularialization */
	    aResiduals[N_LANDMARKS * 2]   = _Tp(m_dRotWeight) * aExtParams[0];
		aResiduals[N_LANDMARKS * 2 + 1] = _Tp(m_dRotWeight) * aExtParams[1];
		aResiduals[N_LANDMARKS * 2 + 2] = _Tp(m_dRotWeight) * aExtParams[2];
		aResiduals[N_LANDMARKS * 2 + 3] = _Tp(m_dTranWeight) * aExtParams[3];
		aResiduals[N_LANDMARKS * 2 + 4] = _Tp(m_dTranWeight) * aExtParams[4];
		aResiduals[N_LANDMARKS * 2 + 5] = _Tp(m_dTranWeight) * aExtParams[5];
		// print_array(aResiduals+N_LANDMARK*2, 6);
		
		return true;
	}

	static ceres::CostFunction *create(dlib::full_object_detection *observedPoints, BfmManager *model, std::vector<unsigned int> aLandmarkMap) 
	{
		return (new ceres::AutoDiffCostFunction<LinearizedExtParamsReprojErr, N_LANDMARKS * 2 + N_EXT_PARAMS, N_EXT_PARAMS>(
			new LinearizedExtParamsReprojErr(observedPoints, model, aLandmarkMap)));
	}

	static ceres::CostFunction *create(dlib::full_object_detection *observedPoints, BfmManager *model, std::vector<unsigned int> aLandmarkMap, double a, double b) 
	{
		return (new ceres::AutoDiffCostFunction<LinearizedExtParamsReprojErr, N_LANDMARKS * 2 + N_EXT_PARAMS, N_EXT_PARAMS>(
			new LinearizedExtParamsReprojErr(observedPoints, model, aLandmarkMap, a, b)));
	}


private:
	dlib::full_object_detection *m_pObservedPoints;
    BfmManager *m_pModel;
	std::vector<unsigned int> m_aLandmarkMap;
	/* residual coefficients */
	double m_dRotWeight = 1.0;	/* for rotation */
	double m_dTranWeight = 1.0;	/* for translation */
};


#endif // HPE_EXT_PARAMS_REPROJ_ERR_H