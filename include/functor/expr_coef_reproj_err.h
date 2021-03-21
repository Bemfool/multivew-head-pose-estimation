#ifndef HPE_EXPR_COEF_REPROJ_ERR_H
#define HPE_EXPR_COEF_REPROJ_ERR_H

#include "bfm_manager.h"
#include "ceres/ceres.h"
#include "db_params.h"

using FullObjectDetection = dlib::full_object_detection;
using DetPairVector = std::vector<std::pair<FullObjectDetection, int>>;
using Eigen::Matrix;
using ceres::CostFunction;
using ceres::AutoDiffCostFunction;


class MultiExprCoefReprojErr {
public:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
	MultiExprCoefReprojErr(
		const DetPairVector &aObjDetections, 
		BaselFaceModelManager *model, 
		DataManager* pDataManager) : 
		m_aObjDetections(aObjDetections), 
		m_pModel(model), 
		m_pDataManager(pDataManager) { }
	
    template<typename _Tp>
	bool operator () (const _Tp* const aExprCoef, _Tp* aResiduals) const {
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
		const double* pdExtParams = m_pModel->getExtParams();
		pExtParams = new _Tp[N_EXT_PARAMS];
		for(auto i = 0u; i < N_EXT_PARAMS; i++) 
			pExtParams[i] = static_cast<_Tp>(pdExtParams[i]);
		scale = static_cast<_Tp>(m_pModel->getMutableScale());

		// Generate blendshape with expression coefficients and transform 
		const Matrix<_Tp, Dynamic, 1> vecPts = m_pModel->genLandmarkBlendshapeByExpr(aExprCoef) * scale;  
		const Matrix<_Tp, Dynamic, 1> vecPtsMC = bfm_utils::TransPoints(pExtParams, vecPts);

		// Compute Euler Distance between landmark and transformed model points in every photos
		for(auto i = 0u; i < len; i++)
		{
			iFace = m_aObjDetections[i].second;
 			const Matrix<_Tp, 4, 4> matCam = aMatCams[iFace].template cast<_Tp>();
			const Matrix<_Tp, Dynamic, 1> vecPtsWC = bfm_utils::TransPoints(matCam, vecPtsMC);
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

		// Regularization
		for(auto i = 0u; i < N_EXPR_PCS; i++)
			aResiduals[N_RES + i] = static_cast<_Tp>(m_dWeight) * aExprCoef[i];


		return true;
	}

	static CostFunction *create(
		const DetPairVector &aObjDetections, 
		BaselFaceModelManager* model, 
		DataManager* pDataManager) 
	{
		return (new AutoDiffCostFunction<MultiExprCoefReprojErr, N_RES + N_EXPR_PCS, N_EXPR_PCS>(
			new MultiExprCoefReprojErr(aObjDetections, model, pDataManager)));
	}

private:
	DetPairVector m_aObjDetections;
    BaselFaceModelManager *m_pModel;
	DataManager* m_pDataManager;
	double* m_pExtParams;
	double m_dWeight = 0.001;
};


class ExprCoefReprojErr {
public:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
	ExprCoefReprojErr(FullObjectDetection *observedPoints, BaselFaceModelManager *model, std::vector<unsigned int> &aLandmarkMap) 
	: m_pObservedPoints(observedPoints), m_pModel(model), m_aLandmarkMap(aLandmarkMap) { }
	
    template<typename _Tp>
	bool operator () (const _Tp* const aExprCoef, _Tp* aResiduals) const {
		_Tp fx = _Tp(m_pModel->getFx()), fy = _Tp(m_pModel->getFy());
		_Tp cx = _Tp(m_pModel->getCx()), cy = _Tp(m_pModel->getCy());

		const Matrix<_Tp, Dynamic, 1> vecLandmarkBlendshape = m_pModel->genLandmarkBlendshapeByExpr(aExprCoef);  

		const double *daExtParams = m_pModel->getExtParams();
        _Tp *taExtParams = new _Tp[N_EXT_PARAMS];
        for(unsigned int iParam = 0; iParam < N_EXT_PARAMS; iParam++)
            taExtParams[iParam] = (_Tp)(daExtParams[iParam]);

		const Matrix<_Tp, Dynamic, 1> vecLandmarkBlendshapeTransformed = bfm_utils::TransPoints(taExtParams, vecLandmarkBlendshape);

		for(int iLandmark = 0; iLandmark < N_LANDMARKS; iLandmark++) 
		{
			unsigned int iDlibLandmarkIdx = m_aLandmarkMap[iLandmark];
			_Tp u = fx * vecLandmarkBlendshapeTransformed(iLandmark * 3) / vecLandmarkBlendshapeTransformed(iLandmark * 3 + 2) + cx;
			_Tp v = fy * vecLandmarkBlendshapeTransformed(iLandmark * 3 + 1) / vecLandmarkBlendshapeTransformed(iLandmark * 3 + 2) + cy;
			aResiduals[iLandmark * 2] = _Tp(m_pObservedPoints->part(iDlibLandmarkIdx).x()) - u;
			aResiduals[iLandmark * 2 + 1] = _Tp(m_pObservedPoints->part(iDlibLandmarkIdx).y()) - v;
		}

		return true;
	}

	static ceres::CostFunction *create(FullObjectDetection *observedPoints, BaselFaceModelManager *model, std::vector<unsigned int> aLandmarkMap) {
		return (new ceres::AutoDiffCostFunction<ExprCoefReprojErr, N_LANDMARKS * 2, N_EXPR_PCS>(
			new ExprCoefReprojErr(observedPoints, model, aLandmarkMap)));
	}

private:
	FullObjectDetection *m_pObservedPoints;
	BaselFaceModelManager *m_pModel;
	std::vector<unsigned int> m_aLandmarkMap;
};


#endif // HPE_EXPR_COEF_REPROJ_ERR_H