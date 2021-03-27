#ifndef HPE_POINT_REPROJ_ERR_H
#define HPE_POINT_REPROJ_ERR_H

#include "bfm_manager.h"
#include "ceres/ceres.h"
#include "db_params.h"


using Eigen::Dynamic;
using Eigen::Map;
using Eigen::Matrix;


class PointReprojErr {
public:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
	PointReprojErr(
		const DetPairVector &aObjDetections, 
		BfmManager *model, 
		DataManager* pDataManager) : 
		m_aObjDetections(aObjDetections), 
		m_pModel(model), 
		m_pDataManager(pDataManager) { }
	
    template<typename _Tp>
	bool operator () (const _Tp* const aPoints, _Tp* aResiduals) const {
		_Tp fx, fy, cx, cy;
		_Tp u, v, invZ;
        unsigned int iFace;
		auto len = m_aObjDetections.size();
		
		// Fetch intrinsic parameters
		fx = static_cast<_Tp>(m_pModel->getFx());
		fy = static_cast<_Tp>(m_pModel->getFy());
		cx = static_cast<_Tp>(m_pModel->getCx());
		cy = static_cast<_Tp>(m_pModel->getCy());
		
		// Fetch camera matrix array
		const auto& aMatCams = m_pDataManager->getCameraMatrices();
		
		Matrix<_Tp, Dynamic, 1> vecPtsMC;
        vecPtsMC.resize(204);
        for(auto i = 0; i < 204; i++)
            vecPtsMC(i) = aPoints[i];

		// Compute Euler Distance between landmark and transformed model points in every photos
		for(auto i = 0u; i < len; i++)
		{
			iFace = m_aObjDetections[i].second;
 			const Matrix<_Tp, 4, 4> matCam = aMatCams[iFace].template cast<_Tp>();
			const auto vecPtsWC = bfm_utils::TransPoints(matCam, vecPtsMC);
			for(auto j = 0u; j < N_LANDMARKS; j++) 
			{
				invZ = static_cast<_Tp>(1.0) / vecPtsWC(j * 3 + 2);
				u = fx * vecPtsWC(j * 3) * invZ + cx;
				v = fy * vecPtsWC(j * 3 + 1) * invZ + cy;
				aResiduals[i * N_LANDMARKS * 2 + j * 2] = static_cast<_Tp>(m_aObjDetections[i].first.part(j).x()) - u;
				aResiduals[i * N_LANDMARKS * 2 + j * 2 + 1] = static_cast<_Tp>(m_aObjDetections[i].first.part(j).y()) - v;
			}
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
		const DetPairVector &aObjDetections, 
		BfmManager* pModel, 
		DataManager* pDataManager) 
	{
		return (new AutoDiffCostFunction<PointReprojErr, N_RES, 68 * 3>(
			new PointReprojErr(aObjDetections, pModel, pDataManager)));
	}

private:
	DetPairVector m_aObjDetections;
    BfmManager *m_pModel;
	DataManager* m_pDataManager;
};


#endif // HPE_POINT_REPROJ_ERR