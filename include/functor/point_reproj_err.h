#ifndef HPE_POINT_REPROJ_ERR_H
#define HPE_POINT_REPROJ_ERR_H

#include "bfm_manager.h"
#include "ceres/ceres.h"
#include "db_params.h"

#include <memory>


using Eigen::Dynamic;
using Eigen::Map;
using Eigen::Matrix;


class PointReprojErr {
public:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
	PointReprojErr(
		const DetPairVector& aObjDetections, 
		std::shared_ptr<BfmManager>& pBfmManager, 
		std::shared_ptr<DataManager>& pDataManager) : 
		m_aObjDetections(aObjDetections), 
		m_pBfmManager(pBfmManager), 
		m_pDataManager(pDataManager) { }
	
    template<typename _Tp>
	bool operator () (const _Tp* const aPoints, _Tp* aResiduals) const {
		_Tp fx, fy, cx, cy;
		_Tp u, v, invZ;
        unsigned int iView, dlibIdx;
		auto len = m_aObjDetections.size();
		
		// Fetch intrinsic parameters
		fx = static_cast<_Tp>(m_pBfmManager->getFx());
		fy = static_cast<_Tp>(m_pBfmManager->getFy());
		cx = static_cast<_Tp>(m_pBfmManager->getCx());
		cy = static_cast<_Tp>(m_pBfmManager->getCy());

		LOG(INFO) << "Int: " << fx << " " << fy << " " << cx << " " << cy;
		LOG(INFO) << "Size: " << len;
		
		// Fetch camera matrix array
		const auto& aMatCams = m_pDataManager->getCameraMatrices();
		// const auto& mapLandmarkIndices = m_pBfmManager->getMapLandmarkIndices();

		Eigen::Matrix<_Tp, Eigen::Dynamic, 1> vecPtsMC;
        vecPtsMC.resize(204);
        for(auto i = 0; i < 204; i++)
		{
            vecPtsMC(i) = aPoints[i];
		}
		LOG(INFO) << vecPtsMC;

		// Compute Euler Distance between landmark and transformed model points in every photos
		for(auto i = 0u; i < len; ++i)
		{
			iView = m_aObjDetections[i].second;

			LOG(INFO) << "View: " << iView;

 			const Matrix<_Tp, 4, 4> matCam = aMatCams[iView].template cast<_Tp>();

			LOG(INFO) << matCam;

			const auto vecPtsWC = bfm_utils::TransPoints(matCam, vecPtsMC);
			for(auto j = 0u; j < N_LANDMARKS; ++j) 
			{
				invZ = static_cast<_Tp>(1.0) / vecPtsWC(j * 3 + 2);
				u = fx * vecPtsWC(j * 3) * invZ + cx;
				v = fy * vecPtsWC(j * 3 + 1) * invZ + cy;
				aResiduals[i * N_LANDMARKS * 2 + j * 2] = static_cast<_Tp>(m_aObjDetections[i].first.part(j).x()) - u;
				aResiduals[i * N_LANDMARKS * 2 + j * 2 + 1] = static_cast<_Tp>(m_aObjDetections[i].first.part(j).y()) - v;
			
				LOG(INFO) << "uv" << u << v;
				LOG(INFO) << "expect:" << m_aObjDetections[i].first.part(j).x() << " " << m_aObjDetections[i].first.part(j).y();
			}

			std::cin.get();
		}

		// Empty residuals
		for(auto i = len; i < N_PHOTOS; i++)
			for(auto j = 0u; j < N_LANDMARKS; j++)
				aResiduals[i * N_LANDMARKS * 2 + j * 2]
				 = aResiduals[i * N_LANDMARKS * 2 + j * 2 + 1]
				 = static_cast<_Tp>(0.0);

		return true;
	}

	static CostFunction *create(
		const DetPairVector& aObjDetections, 
		std::shared_ptr<BfmManager>& pBfmManager, 
		std::shared_ptr<DataManager>& pDataManager) 
	{
		return (new AutoDiffCostFunction<PointReprojErr, N_RES, 68 * 3>(
			new PointReprojErr(aObjDetections, pBfmManager, pDataManager)));
	}

private:
	DetPairVector m_aObjDetections;
    std::shared_ptr<BfmManager> m_pBfmManager;
	std::shared_ptr<DataManager> m_pDataManager;
};


#endif // HPE_POINT_REPROJ_ERR