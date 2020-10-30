#ifndef DATA_MANAGER
#define DATA_MANAGER

#include "file_utils.h"
#include "model.h"
#include "texture.h"

#include "tinyxml2.h"
#include <Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include <fstream>
#include <string>
#include <iostream>
#include <vector>

using namespace tinyxml2;
using namespace boost::filesystem;

class DataManager
{
public:
	DataManager(const std::string &rootDir) :
		m_pathRootDir(path(rootDir)),
		m_pathLandmarkDir(m_pathRootDir / "face_landmarks"),
		m_pathPhotoDir(m_pathRootDir / "image"),
		m_pathXml(m_pathRootDir / "cam.xml"),
		m_pathModel(m_pathRootDir / "photoscan.ply")
	{
		std::cout << "Init Data Manager" << std::endl;
		m_model = new Model(m_pathModel.string());
		this->loadCamInfo();
		this->loadLandmarks();
		this->loadTextures();

		Eigen::Matrix3f matR = m_matModel.inverse().block(0, 0, 3, 3);
		std::cout << matR << std::endl;

 		Eigen::Vector3f eulerAngles = matR.eulerAngles(0, 2, 1);
		std::cout << "Roll: " << eulerAngles(0) << "("  << (eulerAngles(0) * 180.0 / M_PI) << ")" << std::endl;
		std::cout << "Yaw: " << eulerAngles(1) << "("  << (eulerAngles(1) * 180.0 / M_PI) << ")" << std::endl;
		std::cout << "Pitch: " << eulerAngles(2) << "("  << (eulerAngles(2) * 180.0 / M_PI) << ")" << std::endl;

		for(auto i = 0; i < m_aCameraMatrices.size(); i++)
		{
			Eigen::Matrix4f matCam = m_aCameraMatrices[i];
			std::cout << "Cam: " << i << std::endl;
			Eigen::Matrix3f matR = matCam.block(0, 0, 3, 3);
			Eigen::Vector3f eulerAngles = matR.eulerAngles(0, 2, 1);
			std::cout << "Roll: " << eulerAngles(0) << "("  << (eulerAngles(0) * 180.0 / M_PI) << ")" << std::endl;
			std::cout << "Yaw: " << eulerAngles(1) << "("  << (eulerAngles(1) * 180.0 / M_PI) << ")" << std::endl;
			std::cout << "Pitch: " << eulerAngles(2) << "("  << (eulerAngles(2) * 180.0 / M_PI) << ")" << std::endl;
			Eigen::Vector4f vecT = matCam.col(3);
			std::cout << vecT.transpose() << std::endl;
		}
	}

	void saveLandmarks(unsigned int iPickedFace) const
	{
		std::cout << "save landmark from face " << iPickedFace << std::endl;
		path landmarkFile = m_pathLandmarkDir / (file_utils::Id2Str(iPickedFace) + ".txt");
		rename(landmarkFile, path(m_pathLandmarkDir / (file_utils::Id2Str(iPickedFace)
			+ "_" + std::to_string(glfwGetTime()) + ".backup")));
		std::ofstream out(landmarkFile.string());
		auto landmarkCoords = m_aLandmarkCoordsSets[iPickedFace];
		for (auto it = landmarkCoords.begin(); it != landmarkCoords.end(); it++)
		{
			out << std::scientific << std::setprecision(19) << *it << " " << *(++it) << "\n";
		}
		out.close();
	}

	const Model *getModel() const { return m_model; }

	double getF() const { return m_f; }
	double getCx() const { return m_cx; }
	double getCy() const { return m_cy; }
	double getWidth() const { return m_width; }
	double getHeight() const { return m_height; }
	unsigned int getFaces() const { return m_nFaces; }

	const Eigen::Matrix4f& getMatModel() const { return m_matModel; }

	const std::vector<Eigen::Matrix<float, 4, 4>>& getCameraMatrices() const {return m_aCameraMatrices; }
	const std::vector<Eigen::Matrix<float, 3, 4>>& getProjMatrices() const { return m_aProjMatrices; }
	const std::vector<Eigen::Matrix4f>& getInvTransMatrices() const { return m_aInvTransMatrices; }
	const std::vector<Eigen::Vector3f>& getCamPositions() const { return m_aCamPositions; }
	std::vector<std::vector<float>>& getLandmarkCoordsSets() { return m_aLandmarkCoordsSets; }
	const std::vector<Texture>& getTextures() const { return m_aTextures; }

	path m_pathRootDir;
	path m_pathLandmarkDir;
	path m_pathPhotoDir;
	path m_pathModel;
	path m_pathXml;

	Model *m_model;

	// camera infomation
	double m_f;
	double m_cx;
	double m_cy;
	double m_width;
	double m_height;
	unsigned int m_nFaces;

	Eigen::Matrix4f m_matModel;
	std::vector<Eigen::Matrix<float, 4, 4>> m_aCameraMatrices;
	std::vector<Eigen::Matrix<float, 3, 4>> m_aProjMatrices;
	std::vector<Eigen::Matrix<float, 3, 4>> m_aTransMatrices;
	std::vector<Eigen::Matrix4f> m_aInvTransMatrices;
	std::vector<Eigen::Vector3f> m_aCamPositions;

	std::vector<std::vector<float>> m_aLandmarkCoordsSets;
	std::vector<Texture> m_aTextures;

private:

	int loadCamInfo()
	{
		std::cout << "Load camera information" << std::endl;
		m_matModel = Eigen::Matrix4f::Identity();

		m_aProjMatrices.clear();
		m_aTransMatrices.clear();
		m_aInvTransMatrices.clear();
		m_aCamPositions.clear();

		std::vector<double> sensor_f;
		std::vector<double> sensor_cx;
		std::vector<double> sensor_cy;
		std::vector<double> sensor_w;
		std::vector<double> sensor_h;

		std::vector<int> sensor_idxs;

		int sensorNum;

		XMLDocument doc;
		if(doc.LoadFile(m_pathXml.string().c_str()) != XML_SUCCESS)
		{
			std::cout << "XML load failed." << std::endl;
			return -1;
		}

		XMLElement *root = doc.RootElement();
		XMLElement *chunk = root->FirstChildElement("chunk");  //document->chunk->sensors
		if (chunk != NULL)
		{
			//model transform
			XMLElement *xml_transform = chunk->FirstChildElement("transform");
			if (xml_transform != NULL)
			{
				const char* rotationText = xml_transform->FirstChildElement("rotation")->GetText();
				const char* translationText = xml_transform->FirstChildElement("translation")->GetText();

				char *rotation = const_cast<char *>(rotationText);
				char *splits;
				const char *d = " ";
				splits = strtok(rotation, d);
				int row = 0, col = 0;
				while (splits)
				{
					m_matModel(row, col) = atof(splits);
					col++;
					if (col == 3)
					{
						row++;
						col = 0;
					}
					splits = strtok(NULL, d);
				}

				char *translation = const_cast<char *>(translationText);
				splits = strtok(translation, d);
				for (int i = 0; i < 3; i++)
				{
					m_matModel(i, 3) = atof(splits);
					splits = strtok(NULL, d);
				}
			}


			// sensors
			XMLElement *xml_sensors = chunk->FirstChildElement("sensors");
			if (xml_sensors != NULL)
			{
				int sensorNum = atoi(xml_sensors->Attribute("next_id"));
				cout << "sensor nums: " << sensorNum << endl;

				// intial sensor size
				sensor_f.resize(sensorNum);
				sensor_cx.resize(sensorNum);
				sensor_cy.resize(sensorNum);
				sensor_w.resize(sensorNum);
				sensor_h.resize(sensorNum);

				XMLElement *xml_sensor = xml_sensors->FirstChildElement("sensor");
				for (int i = 0; i < sensorNum; i++)
					while (NULL != xml_sensor)
					{
						int sensorId = atoi(xml_sensor->Attribute("id"));
						//cout<<sensorId<<endl;

						XMLElement *xml_calibration = xml_sensor->FirstChildElement("calibration");
						XMLElement *xml_resolution = xml_calibration->FirstChildElement("resolution");

						XMLElement *xml_f = xml_calibration->FirstChildElement("f");
						XMLElement *xml_cx = xml_calibration->FirstChildElement("cx");
						XMLElement *xml_cy = xml_calibration->FirstChildElement("cy");

						sensor_w[sensorId] = atoi(xml_resolution->Attribute("width"));
						sensor_h[sensorId] = atoi(xml_resolution->Attribute("height"));
						sensor_f[sensorId] = atof(xml_f->GetText());
						sensor_cx[sensorId] = atof(xml_cx->GetText());
						sensor_cy[sensorId] = atof(xml_cy->GetText());

						//distoration parameters
						XMLElement *xml_k1 = xml_calibration->FirstChildElement("k1");
						XMLElement *xml_k2 = xml_calibration->FirstChildElement("k2");
						XMLElement *xml_p1 = xml_calibration->FirstChildElement("p1");
						XMLElement *xml_p2 = xml_calibration->FirstChildElement("p2");
						XMLElement *xml_k3 = xml_calibration->FirstChildElement("k3");

						xml_sensor = xml_sensor->NextSiblingElement("sensor");
					}
			}

			//cameras
			XMLElement *cameras = chunk->FirstChildElement("cameras");
			if (cameras != NULL)
			{
				m_nFaces = atoi(cameras->Attribute("next_id"));
				cout << "camera nums: " << m_nFaces << endl;

				sensor_idxs.resize(m_nFaces);

				XMLElement *xml_camera = cameras->FirstChildElement("camera");
				while (NULL != xml_camera)
				{

					Eigen::Matrix<float, 4, 4> T_camera;
					Eigen::Matrix<float, 3, 4> P;
					int camera_idx = atoi(xml_camera->Attribute("id"));
					int sensor_idx = atoi(xml_camera->Attribute("sensor_id"));
					//cout << "camera " << camera_idx << " using sensor " << sensor_idx << endl;


					const char* transformtmp = xml_camera->FirstChildElement("transform")->GetText();
					char *transform = const_cast<char *>(transformtmp);
					char *splits;
					const char *d = " ";
					splits = strtok(transform, d);
					int row = 0, col = 0;
					while (splits) {
						T_camera(row, col) = atof(splits);
						col++;
						if (col == 4) {
							row++;
							col = 0;
						}
						splits = strtok(NULL, d);

					}

					double f = sensor_f[sensor_idx];
					double cx = sensor_cx[sensor_idx];
					double cy = sensor_cy[sensor_idx];
					double w = sensor_w[sensor_idx];
					double h = sensor_h[sensor_idx];
					Eigen::Matrix<float, 3, 4> K;
					K << f, 0, w / 2.0 + cx, 0,
						0, f, h / 2.0 + cy, 0,
						0, 0, 1, 0;

					Eigen::Matrix<float, 4, 4> T = T_camera.inverse() * m_matModel.inverse();
					P = K * T;

					m_aCameraMatrices.push_back(T_camera.inverse());
					m_aProjMatrices.push_back(P);
					m_aTransMatrices.push_back(T.block(0, 0, 3, 4));
					m_aInvTransMatrices.push_back(T.inverse());

					Eigen::Matrix3f R = T.block(0, 0, 3, 3);
					Eigen::Vector3f t(T(0, 3), T(1, 3), T(2, 3));
					Eigen::Vector3f tmp = -R.transpose()*t;
					m_aCamPositions.push_back(Eigen::Vector3f(tmp[0], tmp[1], tmp[2]));

					sensor_idxs[camera_idx] = sensor_idx;

					xml_camera = xml_camera->NextSiblingElement("camera");
				}
			}
		}

		m_f = sensor_f[0];
		m_cx = sensor_cx[0];
		m_cy = sensor_cy[0];
		m_width = sensor_w[0];
		m_height = sensor_h[0];

		return 0;
	}

	void loadLandmarks()
	{
		std::cout << "Load landmarks" << std::endl;
		m_aLandmarkCoordsSets.resize(m_nFaces);
		path path(m_pathLandmarkDir);

		if (!exists(path))
		{
			std::cout << "Error: Path " << m_pathLandmarkDir << " does not exist." << std::endl;
			return;
		}

		for (directory_iterator iter(m_pathLandmarkDir); iter != directory_iterator(); iter++)
		{
			if (iter->path().extension() != ".txt")
				continue;

			unsigned int camera_id = file_utils::Str2Id(iter->path().stem().string());
			std::string filename = iter->path().string();
			std::cout << "Load landmarks from: " << filename;

			std::ifstream in(filename);
			if (!in)
			{
				std::cout << "\nError: Can not open " << iter->path().string() << "." << std::endl;
				return;
			}
			std::string line;
			while (std::getline(in, line))
			{
				if (line.empty() || line == "\n")
					continue;
				std::vector<std::string> words;
				boost::split(words, line, boost::is_any_of(" "));
				m_aLandmarkCoordsSets[camera_id].push_back(std::stof(words[0]));
				m_aLandmarkCoordsSets[camera_id].push_back(std::stof(words[1]));
			}
			std::cout << " - " << m_aLandmarkCoordsSets[camera_id].size() << std::endl;
		}
	}

	void loadTextures()
	{
		m_aTextures.resize(m_nFaces);
		for (auto iFace = 0; iFace < m_nFaces; iFace++)
		{
			path pathTexture = m_pathPhotoDir / (file_utils::Id2Str(iFace) + ".jpg");
			std::cout << "Load texture: " << pathTexture.string() << std::endl;
			m_aTextures[iFace] = Texture::LoadTexture(pathTexture.string());
			if (!m_aLandmarkCoordsSets[iFace].empty() && 
				m_aTextures[iFace].getWidth() > m_aTextures[iFace].getHeight())
			{
				if (m_aLandmarkCoordsSets[iFace][0] < m_aTextures[iFace].getWidth() * 0.5)
					m_aTextures[iFace].setRotateType(RotateType_CCW);
				else
					m_aTextures[iFace].setRotateType(RotateType_CW);
			}
			else
				m_aTextures[iFace].setRotateType(RotateType_No);
		}
	}
};


#endif // DATA_MANAGER_H