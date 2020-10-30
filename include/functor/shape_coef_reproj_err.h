#ifndef HPE_SHAPE_COEF_REPROJ_ERR_H
#define HPE_SHAPE_COEF_REPROJ_ERR_H

#include "bfm_manager.h"
#include "data_manager.h"
#include "ceres/ceres.h"
#include "db_params.h"

using FullObjectDetection = dlib::full_object_detection;
using Eigen::Matrix;
using ceres::CostFunction;
using ceres::AutoDiffCostFunction;


class MultiShapeCoefReprojErr {
public:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
	MultiShapeCoefReprojErr(
		std::vector<std::pair<dlib::full_object_detection, int>> &aObjDetections, 
		BaselFaceModelManager *model, 
		DataManager* pDataManager,
		double* pExtParams) : 
		m_aObjDetections(aObjDetections), 
		m_pModel(model), 
		m_pDataManager(pDataManager),
		m_pExtParams(pExtParams) { }
	
    template<typename _Tp>
	bool operator () (const _Tp* const aShapeCoef, _Tp* aResiduals) const {
		_Tp fx = _Tp(m_pModel->getFx()), fy = _Tp(m_pModel->getFy());
		_Tp cx = _Tp(m_pModel->getCx()), cy = _Tp(m_pModel->getCy());
		const std::vector<Eigen::Matrix4f>& aMatCams = m_pDataManager->getCameraMatrices();

		const Matrix<_Tp, Dynamic, 1> vecPts = m_pModel->genLandmarkBlendshapeByShape(aShapeCoef);  

		for(unsigned int iPhoto = 0; iPhoto < m_aObjDetections.size(); iPhoto++)
		{
			int idx = m_aObjDetections[iPhoto].second;
			_Tp pExtParams[6] = {
				(_Tp)m_pExtParams[0], (_Tp)m_pExtParams[1], (_Tp)m_pExtParams[2],
				(_Tp)m_pExtParams[3 + idx * 3], (_Tp)m_pExtParams[3 + idx * 3 + 1], (_Tp)m_pExtParams[3 + idx * 3 + 2]
			};

 			const Eigen::Matrix<_Tp, 4, 4> matCam = aMatCams[idx].template cast<_Tp>();

			const Matrix<_Tp, Dynamic, 1> vecPtsMC = bfm_utils::TransPoints(pExtParams, vecPts);
			const Matrix<_Tp, Dynamic, 1> vecPtsWC = bfm_utils::TransPoints(matCam, vecPtsMC);
			for(unsigned int iLandmark = 0; iLandmark < N_LANDMARKS; iLandmark++) 
			{
				_Tp u = fx * vecPtsWC(iLandmark * 3) / vecPtsWC(iLandmark * 3 + 2) + cx;
				_Tp v = fy * vecPtsWC(iLandmark * 3 + 1) / vecPtsWC(iLandmark * 3 + 2) + cy;
				aResiduals[iPhoto * N_LANDMARKS * 2 + iLandmark * 2] = _Tp(m_aObjDetections[iPhoto].first.part(iLandmark).x()) - u;
				aResiduals[iPhoto * N_LANDMARKS * 2 + iLandmark * 2 + 1] = _Tp(m_aObjDetections[iPhoto].first.part(iLandmark).y()) - v;
			}
		}

		for(unsigned int iPhoto = m_aObjDetections.size();+ iPhoto < N_PHOTOS; iPhoto++)
		{
			for(unsigned int iLandmark = 0; iLandmark < N_LANDMARKS; iLandmark++)
			{
				aResiduals[iPhoto * N_LANDMARKS * 2 + iLandmark * 2] = _Tp(0.0);
				aResiduals[iPhoto * N_LANDMARKS * 2 + iLandmark * 2 + 1] = _Tp(0.0);
			}
		}

		for(auto i = 0; i < N_ID_PCS; i++)
			aResiduals[N_LANDMARKS * 2 * N_PHOTOS + i] = _Tp(m_dWeight) * aShapeCoef[i];

		return true;
	}

	static CostFunction *create(
		std::vector<std::pair<dlib::full_object_detection, int>> &aObjDetections, 
		BaselFaceModelManager* model, 
		DataManager* pDataManager,
		double* pExtParams) 
	{
		return (new AutoDiffCostFunction<MultiShapeCoefReprojErr, N_LANDMARKS * 2 * N_PHOTOS + N_ID_PCS, N_ID_PCS>(
			new MultiShapeCoefReprojErr(aObjDetections, model, pDataManager, pExtParams)));
	}

private:
	std::vector<std::pair<dlib::full_object_detection, int>> m_aObjDetections;
    BaselFaceModelManager *m_pModel;
	DataManager* m_pDataManager;
	double* m_pExtParams;
	double m_dWeight = 300.0;
};


class ShapeCoefReprojErr {
public:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
	ShapeCoefReprojErr(FullObjectDetection *observedPoints, BaselFaceModelManager *model, std::vector<unsigned int> aLandmarkMap) 
		: m_pObservedPoints(observedPoints), m_pModel(model), m_aLandmarkMap(aLandmarkMap) { }
	
    template<typename _Tp>
	bool operator () (const _Tp* const aShapeCoef, _Tp* aResiduals) const {
		_Tp fx = _Tp(m_pModel->getFx()), fy = _Tp(m_pModel->getFy());
		_Tp cx = _Tp(m_pModel->getCx()), cy = _Tp(m_pModel->getCy());
		
		const Matrix<_Tp, Dynamic, 1> vecLandmarkBlendshape = m_pModel->genLandmarkBlendshapeByShape(aShapeCoef);  

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

	static CostFunction *create(FullObjectDetection *observedPoints, BaselFaceModelManager *model, std::vector<unsigned int> aLandmarkMap) {
		return (new AutoDiffCostFunction<ShapeCoefReprojErr, N_LANDMARKS * 2, N_ID_PCS>(
			new ShapeCoefReprojErr(observedPoints, model, aLandmarkMap)));
	}

private:
	FullObjectDetection *m_pObservedPoints;
	BaselFaceModelManager *m_pModel;
	std::vector<unsigned int> m_aLandmarkMap;
};


#endif // HPE_SHAPE_COEF_REPROJ_ERR_H