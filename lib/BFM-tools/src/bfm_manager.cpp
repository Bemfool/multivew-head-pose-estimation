﻿#include "bfm_manager.h"


BaselFaceModelManager::BaselFaceModelManager(
	std::string strModelPath,
	unsigned int nVertices,
	unsigned int nFaces,
	unsigned int nIdPcs,
	unsigned int nExprPcs,
	double *aIntParams, 
	std::string strShapeMuH5Path,
	std::string strShapeEvH5Path,
	std::string strShapePcH5Path,
	std::string strTexMuH5Path,
	std::string strTexEvH5Path,
	std::string strTexPcH5Path,
	std::string strExprMuH5Path,
	std::string strExprEvH5Path,
	std::string strExprPcH5Path,
	std::string strTriangleListH5Path,
	unsigned int nLandmarks,
	std::string strLandmarkIdxPath) :
	m_strModelPath(strModelPath),
	m_nVertices(nVertices),
	m_nFaces(nFaces),
	m_nIdPcs(nIdPcs),
	m_nExprPcs(nExprPcs),
	m_nLandmarks(nLandmarks),
	m_strLandmarkIdxPath(strLandmarkIdxPath),
	m_strShapeMuH5Path(strShapeMuH5Path),
	m_strShapeEvH5Path(strShapeEvH5Path),
	m_strShapePcH5Path(strShapePcH5Path),
	m_strTexMuH5Path(strTexMuH5Path),
	m_strTexEvH5Path(strTexEvH5Path),
	m_strTexPcH5Path(strTexPcH5Path),
	m_strExprMuH5Path(strExprMuH5Path),
	m_strExprEvH5Path(strExprEvH5Path),
	m_strExprPcH5Path(strExprPcH5Path),
	m_strTriangleListH5Path(strTriangleListH5Path) 
{
	for(unsigned int iParam = 0; iParam < 4; iParam++)
		m_aIntParams[iParam] = aIntParams[iParam];

	m_bUseLandmark = nLandmarks == 0 ? false : true;

	this->alloc();
	this->load();
	
	unsigned int iTex = 0;
	while(m_bIsTexStd)
	{
		if(m_vecTexMu(iTex++) > 1.0)
			m_bIsTexStd = false;
	}

	if(m_bUseLandmark)
	{
		this->extractLandmarks();
		this->genLandmarkBlendshape();
	}
	this->genAvgFace();
}


void BaselFaceModelManager::alloc() {
	m_aShapeCoef = new double[m_nIdPcs];
	std::fill(m_aShapeCoef, m_aShapeCoef + m_nIdPcs, 0.0);
	m_vecShapeMu.resize(m_nVertices * 3);
	m_vecShapeEv.resize(m_nIdPcs);
	m_matShapePc.resize(m_nVertices * 3, m_nIdPcs);

	m_aTexCoef = new double[m_nIdPcs];
	std::fill(m_aTexCoef, m_aTexCoef + m_nIdPcs, 0.0);
	m_vecTexMu.resize(m_nVertices * 3);
	m_vecTexEv.resize(m_nIdPcs);
	m_matTexPc.resize(m_nVertices * 3, m_nIdPcs);

	m_aExprCoef = new double[m_nExprPcs];
	std::fill(m_aExprCoef, m_aExprCoef + m_nExprPcs, 0.0);
	m_vecExprMu.resize(m_nVertices * 3);
	m_vecExprEv.resize(m_nExprPcs);
	m_matExprPc.resize(m_nVertices * 3, m_nExprPcs);

	m_vecTriangleList.resize(m_nFaces * 3);

	m_vecCurrentShape.resize(m_nVertices * 3);
	m_vecCurrentTex.resize(m_nVertices * 3);
	m_vecCurrentExpr.resize(m_nVertices * 3);
	m_vecCurrentBlendshape.resize(m_nVertices * 3);

	if (m_bUseLandmark) {
		m_vecLandmarkIndices.resize(m_nLandmarks);
		m_vecLandmarkShapeMu.resize(m_nLandmarks * 3);
		m_matLandmarkShapePc.resize(m_nLandmarks * 3, m_nIdPcs);
		m_vecLandmarkExprMu.resize(m_nLandmarks * 3);
		m_matLandmarkExprPc.resize(m_nLandmarks * 3, m_nExprPcs);
	}
}


bool BaselFaceModelManager::load() {
	float *vecShapeMu = new float[m_nVertices * 3];
	float *vecShapeEv = new float[m_nIdPcs];
	float *matShapePc = new float[m_nVertices * 3 * m_nIdPcs];
	float *vecTexMu = new float[m_nVertices * 3];
	float *vecTexEv = new float[m_nIdPcs];
	float *matTexPc = new float[m_nVertices * 3 * m_nIdPcs];
	float *vecExprMu = new float[m_nVertices * 3];
	float *vecExprEv = new float[m_nExprPcs];
	float *matExprPc = new float[m_nVertices * 3 * m_nExprPcs];
	unsigned int *vecTriangleList = new unsigned int[m_nFaces * 3];

	hid_t file = H5Fopen(m_strModelPath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

	bfm_utils::LoadH5Model(file, m_strShapeMuH5Path, vecShapeMu, m_vecShapeMu, H5T_NATIVE_FLOAT);
	bfm_utils::LoadH5Model(file, m_strShapeEvH5Path, vecShapeEv, m_vecShapeEv, H5T_NATIVE_FLOAT);
	bfm_utils::LoadH5Model(file, m_strShapePcH5Path, matShapePc, m_matShapePc, H5T_NATIVE_FLOAT);

	bfm_utils::LoadH5Model(file, m_strTexMuH5Path, vecTexMu, m_vecTexMu, H5T_NATIVE_FLOAT);
	bfm_utils::LoadH5Model(file, m_strTexEvH5Path, vecTexEv, m_vecTexEv, H5T_NATIVE_FLOAT);
	bfm_utils::LoadH5Model(file, m_strTexPcH5Path, matTexPc, m_matTexPc, H5T_NATIVE_FLOAT);
	
	bfm_utils::LoadH5Model(file, m_strExprMuH5Path, vecExprMu, m_vecExprMu, H5T_NATIVE_FLOAT);
	bfm_utils::LoadH5Model(file, m_strExprEvH5Path, vecExprEv, m_vecExprEv, H5T_NATIVE_FLOAT);
	bfm_utils::LoadH5Model(file, m_strExprPcH5Path, matExprPc, m_matExprPc, H5T_NATIVE_FLOAT);
	
	bfm_utils::LoadH5Model(file, m_strTriangleListH5Path, vecTriangleList, m_vecTriangleList, H5T_NATIVE_UINT32);

	std::cout << "status: " << H5Fclose(file) << std::endl;
	// m_vecShapeMu = m_vecShapeMu * 1000.0;
	m_vecShapeMu = m_vecShapeMu * 0.01;
	m_vecExprMu = m_vecExprMu * 0.01;
	std::cout << m_vecShapeMu(0) << " " << m_vecShapeMu(1) << " " << m_vecShapeMu(2) << std::endl;

	if(m_bUseLandmark)
	{
		ifstream in(m_strLandmarkIdxPath, std::ios::in);
		if (!in) 
		{
			BFM_DEBUG("[ERROR] Can't open %s.", m_strLandmarkIdxPath.c_str());
			return false;
		}

		unsigned int iLandmark;
		for (unsigned int i = 0; i < m_nLandmarks; i++) 
		{
			in >> iLandmark;
			m_vecLandmarkIndices[i] = iLandmark - 1;
		}

		in.close();
	}
	
	return true;
}


void BaselFaceModelManager::extractLandmarks() 
{
	for(unsigned int iLandmark = 0; iLandmark < m_nLandmarks; iLandmark++) 
	{
		unsigned int idx = m_vecLandmarkIndices[iLandmark];
		m_vecLandmarkShapeMu(iLandmark * 3) = m_vecShapeMu(idx * 3);
		m_vecLandmarkShapeMu(iLandmark * 3 + 1) = m_vecShapeMu(idx * 3 + 1);
		m_vecLandmarkShapeMu(iLandmark * 3 + 2) = m_vecShapeMu(idx * 3 + 2);
		m_vecLandmarkExprMu(iLandmark * 3) = m_vecExprMu(idx * 3);
		m_vecLandmarkExprMu(iLandmark * 3 + 1) = m_vecExprMu(idx * 3 + 1);
		m_vecLandmarkExprMu(iLandmark * 3 + 2) = m_vecExprMu(idx * 3 + 2);

		for(unsigned int iIdPc = 0; iIdPc < m_nIdPcs; iIdPc++) 
		{
			m_matLandmarkShapePc(iLandmark * 3, iIdPc) = m_matShapePc(idx * 3, iIdPc);
			m_matLandmarkShapePc(iLandmark * 3 + 1, iIdPc) = m_matShapePc(idx * 3 + 1, iIdPc);
			m_matLandmarkShapePc(iLandmark * 3 + 2, iIdPc) = m_matShapePc(idx * 3 + 2, iIdPc);	
		}

		for(unsigned int iExprPc = 0; iExprPc < m_nExprPcs; iExprPc++) 
		{
			m_matLandmarkExprPc(iLandmark * 3, iExprPc) = m_matExprPc(idx * 3, iExprPc);
			m_matLandmarkExprPc(iLandmark * 3 + 1, iExprPc) = m_matExprPc(idx * 3 + 1, iExprPc);
			m_matLandmarkExprPc(iLandmark * 3 + 2, iExprPc) = m_matExprPc(idx * 3 + 2, iExprPc);
		}
	}
}


void BaselFaceModelManager::genRndFace(double dScale) 
{
	if(dScale == 0.0)
	{
		BFM_DEBUG("[BFM_MANAGER] Generate average face\n");
	}
	else
	{
		BFM_DEBUG("[BFM_MANAGER] Generate random face (using the same scale)\n");
	}
	m_aShapeCoef = bfm_utils::randn(m_nIdPcs, dScale);
	m_aTexCoef   = bfm_utils::randn(m_nIdPcs, dScale);
	m_aExprCoef  = bfm_utils::randn(m_nExprPcs, dScale);
	this->genFace();
}


void BaselFaceModelManager::genRndFace(double dShapeScale, double dTexScale, double dExprScale) 
{
	BFM_DEBUG("[BFM_MANAGER] Generate random face (using different scales)\n");
	m_aShapeCoef = bfm_utils::randn(m_nIdPcs, dShapeScale);
	m_aTexCoef   = bfm_utils::randn(m_nIdPcs, dTexScale);
	m_aExprCoef  = bfm_utils::randn(m_nExprPcs, dExprScale);
	this->genFace();
}


void BaselFaceModelManager::genFace() 
{
	BFM_DEBUG("[BFM_MANAGER]Generate face with shape and expression coefficients -");

	m_vecCurrentShape = this->coef2Object(m_aShapeCoef, m_vecShapeMu, m_matShapePc, m_vecShapeEv, m_nIdPcs);
	m_vecCurrentTex   = this->coef2Object(m_aTexCoef, m_vecTexMu, m_matTexPc, m_vecTexEv, m_nIdPcs);
	m_vecCurrentExpr  = this->coef2Object(m_aExprCoef, m_vecExprMu, m_matExprPc, m_vecExprEv, m_nExprPcs);
	m_vecCurrentBlendshape = m_vecCurrentShape + m_vecCurrentExpr;

	BFM_DEBUG("Success\n");
}


void BaselFaceModelManager::genLandmarkBlendshape()  
{
	BFM_DEBUG("[BFM_MANAGER] Generate landmarks with shape and expression coefficients -");

	m_vecLandmarkCurrentShape = this->coef2Object(m_aShapeCoef, m_vecLandmarkShapeMu, m_matLandmarkShapePc, m_vecShapeEv, m_nIdPcs);
	m_vecLandmarkCurrentExpr = this->coef2Object(m_aExprCoef, m_vecLandmarkExprMu, m_matLandmarkExprPc, m_vecExprEv, m_nExprPcs);
	m_vecLandmarkCurrentBlendshape = m_vecLandmarkCurrentShape + m_vecLandmarkCurrentExpr;

	BFM_DEBUG("Success\n");
}


void BaselFaceModelManager::genRMat() 
{
	BFM_DEBUG("[BFM_MANAGER] Generate rotation matrix.\n");
	const double &roll   = m_aExtParams[0];
	const double &yaw    = m_aExtParams[1];
	const double &pitch  = m_aExtParams[2];
	m_matR = bfm_utils::Euler2Mat(roll, yaw, pitch, false);
}


void BaselFaceModelManager::genTVec()
{
	BFM_DEBUG("[BFM_MANAGER] Generate translation vector.\n");	
	const double &tx = m_aExtParams[3];
	const double &ty = m_aExtParams[4];
	const double &tz = m_aExtParams[5];
	m_vecT << tx, ty, tz;	
}


void BaselFaceModelManager::genTransMat()
{
	this->genRMat();
	this->genTVec();
}


void BaselFaceModelManager::genExtParams()
{
	BFM_DEBUG("generate external paramter:\n");

	if(!bfm_utils::IsRMat(m_matR))
	{
		BFM_DEBUG("	detect current matrix does not satisfy constraints - ");
		bfm_utils::SatisfyExtMat(m_matR, m_vecT);
		BFM_DEBUG("solve\n");
	}

	double sy = std::sqrt(m_matR(0,0) * m_matR(0,0) +  m_matR(1,0) * m_matR(1,0));
    bool bIsSingular = sy < 1e-6;

    if (!bIsSingular) 
	{
        m_aExtParams[2] = atan2(m_matR(2,1) , m_matR(2,2));
        m_aExtParams[1] = atan2(-m_matR(2,0), sy);
        m_aExtParams[0] = atan2(m_matR(1,0), m_matR(0,0));
    } 
	else 
	{
        m_aExtParams[2] = atan2(-m_matR(1,2), m_matR(1,1));
        m_aExtParams[1] = atan2(-m_matR(2,0), sy);
        m_aExtParams[0] = 0;
    }
	m_aExtParams[3] = m_vecT(0, 0);
	m_aExtParams[4] = m_vecT(1, 0);
	m_aExtParams[5] = m_vecT(2, 0);
	
	this->genTransMat();
}


void BaselFaceModelManager::accExtParams(double *aExtParams) 
{
	/* in every iteration, P = R`(RP+t)+t`, 
	 * R_{new} = R`R_{old}
	 * t_{new} = R`t_{old} + t`
	 */

	Matrix3d matR;
	Vector3d vecT;	
	double dYaw   = aExtParams[0];
	double dPitch = aExtParams[1];
	double dRoll  = aExtParams[2];
	double dTx = aExtParams[3];
	double dTy = aExtParams[4];
	double dTz = aExtParams[5];

	/* accumulate rotation */
	matR = bfm_utils::Euler2Mat(dYaw, dPitch, dRoll, true);
	m_matR = matR * m_matR;

	/* accumulate translation */
	vecT << dTx, dTy, dTz;
	m_vecT = matR * m_vecT + vecT;
}	


void BaselFaceModelManager::writePly(std::string fn, long mode) const 
{
	std::ofstream out;
	/* Note: In Linux Cpp, we should use std::ios::BFM_OUT as flag, which is not necessary in Windows */
	out.open(fn, std::ios::out | std::ios::binary);
	if (!out) 
	{
		BFM_DEBUG("Creation of %s failed.\n", fn.c_str());
		return;
	}
	out << "ply\n";
	out << "format binary_little_endian 1.0\n";
	out << "comment Made from the 3D Morphable Face Model of the Univeristy of Basel, Switzerland.\n";
	out << "element vertex " << m_nVertices << "\n";
	out << "property float x\n";
	out << "property float y\n";
	out << "property float z\n";
	out << "property uchar red\n";
	out << "property uchar green\n";
	out << "property uchar blue\n";
	out << "element face " << m_nFaces << "\n";
	out << "property list uchar int vertex_indices\n";
	out << "end_header\n";

	int cnt = 0;
	for (int iVertice = 0; iVertice < m_nVertices; iVertice++) 
	{
		float x, y, z;
		if(mode & ModelWriteMode_NoExpr) 
		{
			x = float(m_vecCurrentShape(iVertice * 3)) ;
			y = float(m_vecCurrentShape(iVertice * 3 + 1));
			z = float(m_vecCurrentShape(iVertice * 3 + 2));
		} 
		else 
		{
			// x = float(m_vecShapeMu(iVertice * 3));
			// y = float(m_vecShapeMu(iVertice * 3 + 1));
			// z = float(m_vecShapeMu(iVertice * 3 + 2));
			x = float(m_vecCurrentBlendshape(iVertice * 3));
			y = float(m_vecCurrentBlendshape(iVertice * 3 + 1));
			z = float(m_vecCurrentBlendshape(iVertice * 3 + 2));
		}

		if(mode & ModelWriteMode_CameraCoord) 
		{
			bfm_utils::Trans(m_aExtParams, x, y, z);
			// y = -y; z = -z;
		}

		unsigned char r, g, b;
		if ((mode & ModelWriteMode_PickLandmark) && 
			std::find(m_vecLandmarkIndices.begin(), m_vecLandmarkIndices.end(), iVertice) != m_vecLandmarkIndices.end()) 
		{
			r = 255;
			g = 0;
			b = 0;
			cnt++;
		} 
		else 
		{
			r = m_vecCurrentTex(iVertice * 3);
			g = m_vecCurrentTex(iVertice * 3 + 1);
			b = m_vecCurrentTex(iVertice * 3 + 2);
		}

		out.write((char *)&x, sizeof(x));
		out.write((char *)&y, sizeof(y));
		out.write((char *)&z, sizeof(z));
		out.write((char *)&r, sizeof(r));
		out.write((char *)&g, sizeof(g));
		out.write((char *)&b, sizeof(b));
	}

	if ((mode & ModelWriteMode_PickLandmark) && cnt != m_nLandmarks) 
	{
		BFM_DEBUG("[ERROR] Pick too less landmarks.\n");
		BFM_DEBUG("Number of picked points is %d.\n", cnt);
	}

	unsigned char N_VER_PER_FACE = 3;
	for (int iFace = 0; iFace < m_nFaces; iFace++) 
	{
		out.write((char *)&N_VER_PER_FACE, sizeof(N_VER_PER_FACE));
		int x = m_vecTriangleList(iFace * 3) - 1;
		int y = m_vecTriangleList(iFace * 3 + 1) - 1;
		int z = m_vecTriangleList(iFace * 3 + 2) - 1;
		out.write((char *)&y, sizeof(y));
		out.write((char *)&x, sizeof(x));
		out.write((char *)&z, sizeof(z));
	}

	out.close();
}


void BaselFaceModelManager::writeLandmarkPly(std::string fn) const {
	std::ofstream out;
	/* Note: In Linux Cpp, we should use std::ios::BFM_OUT as flag, which is not necessary in Windows */
	out.open(fn, std::ios::out | std::ios::binary);
	if (!out) 
	{
		BFM_DEBUG("Creation of %s failed.\n", fn.c_str());
		return;
	}

	out << "ply\n";
	out << "format binary_little_endian 1.0\n";
	out << "comment Made from the 3D Morphable Face Model of the Univeristy of Basel, Switzerland.\n";
	out << "element vertex " << m_nLandmarks << "\n";
	out << "property float x\n";
	out << "property float y\n";
	out << "property float z\n";
	out << "end_header\n";

	int cnt = 0;
	for (int i = 0; i < m_nLandmarks; i++) 
	{
		float x, y, z;
		x = float(m_vecLandmarkCurrentBlendshape(i * 3));
		y = float(m_vecLandmarkCurrentBlendshape(i * 3 + 1));
		z = float(m_vecLandmarkCurrentBlendshape(i * 3 + 2));
		out.write((char *)&x, sizeof(x));
		out.write((char *)&y, sizeof(y));
		out.write((char *)&z, sizeof(z));
	}

	out.close();	
}


void BaselFaceModelManager::clrExtParams()
{
	std::fill(m_aExtParams, m_aExtParams + 6, 0.0);
	this->genTransMat();
	this->genFace();
}
