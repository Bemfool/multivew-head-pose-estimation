#ifndef HPE_SHAPE_COEF_REPROJ_ERR_H
#define HPE_SHAPE_COEF_REPROJ_ERR_H

#include "bfm_manager.h"
#include "data_manager.h"
#include "ceres/ceres.h"
#include "db_params.h"

using FullObjectDetection = dlib::full_object_detection;
using DetPairVector = std::vector<std::pair<FullObjectDetection, int>>;
using Eigen::Matrix;
using ceres::CostFunction;
using ceres::AutoDiffCostFunction;


class MultiShapeCoefReprojErr {
public:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
	MultiShapeCoefReprojErr(
		const DetPairVector &aObjDetections, 
		BfmManager *model, 
		DataManager* pDataManager,
		double dWeight,
		std::vector<std::vector<bool>>& validList) : 
		m_aObjDetections(aObjDetections), 
		m_pModel(model), 
		m_pDataManager(pDataManager),
		m_dWeight(dWeight),
		m_validList(validList) { }
	
    template<typename _Tp>
	bool operator () (const _Tp* const aShapeCoef, _Tp* aResiduals) const {
		_Tp fx, fy, cx, cy;
		_Tp *pExtParams;
		_Tp scale;
		_Tp u, v, invZ;
		int iFace;
		auto len = m_aObjDetections.size();
		
		// Fetch intrinsic parameters
		fx = static_cast<_Tp>(m_pModel->getFx());
		fy = static_cast<_Tp>(m_pModel->getFy());
		cx = static_cast<_Tp>(m_pModel->getCx());
		cy = static_cast<_Tp>(m_pModel->getCy());
		
		// Fetch camera matrix array
		const auto& aMatCams = m_pDataManager->getCameraMatrices();
		
		// Fetch extrinsic parameters and scale
		const double* pdExtParams = m_pModel->getExtParams().data();
		pExtParams = new _Tp[N_EXT_PARAMS];
		for(auto i = 0u; i < N_EXT_PARAMS; i++) 
			pExtParams[i] = static_cast<_Tp>(pdExtParams[i]);
		scale = static_cast<_Tp>(m_pModel->getMutableScale());

		// Generate blendshape with shape coefficients and transform 
		const Matrix<_Tp, Dynamic, 1> vecPts = m_pModel->genLandmarkBlendshapeByShape(aShapeCoef) * scale;  
		const Matrix<_Tp, Dynamic, 1> vecPtsMC = bfm_utils::TransPoints(pExtParams, vecPts);

		// Compute Euler Distance between landmark and transformed model points in every photos
		for(auto i = 0u; i < len; i++)
		{
			iFace = m_aObjDetections[i].second;
 			const Matrix<_Tp, 4, 4> matCam = aMatCams[iFace].template cast<_Tp>();
			const auto vecPtsWC = bfm_utils::TransPoints(matCam, vecPtsMC);
			for(auto j = 0u; j < N_LANDMARKS; j++) 
			{

				if(m_validList[i][j])
				{
					invZ = static_cast<_Tp>(1.0) / vecPtsWC(j * 3 + 2);
					u = fx * vecPtsWC(j * 3) * invZ + cx;
					v = fy * vecPtsWC(j * 3 + 1) * invZ + cy;
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
		for(auto i = len; i < N_PHOTOS; i++)
			for(auto j = 0u; j < N_LANDMARKS; j++)
				aResiduals[i * N_LANDMARKS * 2 + j * 2]
				 = aResiduals[i * N_LANDMARKS * 2 + j * 2 + 1]
				 = static_cast<_Tp>(0.0);

		// Regularization
		for(auto i = 0u; i < N_ID_PCS; i++)
			aResiduals[N_RES + i] = static_cast<_Tp>(m_dWeight) * aShapeCoef[i];

		delete[] pExtParams;

		return true;
	}

	static CostFunction *create(
		const DetPairVector &aObjDetections, 
		BfmManager* pModel, 
		DataManager* pDataManager,
		double dWeight,
		std::vector<std::vector<bool>>& validList) 
	{
		return (new AutoDiffCostFunction<MultiShapeCoefReprojErr, N_RES + N_ID_PCS, N_ID_PCS>(
			new MultiShapeCoefReprojErr(aObjDetections, pModel, pDataManager, dWeight, validList)));
	}

private:
	DetPairVector m_aObjDetections;
    BfmManager *m_pModel;
	DataManager* m_pDataManager;
	double m_dWeight;
	std::vector<std::vector<bool>>& m_validList;
};



#endif // HPE_SHAPE_COEF_REPROJ_ERR_H