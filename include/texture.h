#ifndef TEXTURE_H
#define TEXTURE_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>

#include <string>
#include <iostream>

enum RotateType
{
	RotateType_No = 1,
	RotateType_CW = 2,
	RotateType_CCW = 3
};


class Texture
{
public:
	Texture() { }
 	Texture(unsigned int id, std::string path, int width, int height)
	  : m_id(id), m_path(path), m_width(width), m_height(height) { }
	void setRotateType(RotateType rotateType) { m_rotateType = rotateType; }
	RotateType getRotateType() const { return m_rotateType; }
	int getWidth() const { return m_width; }
	int getHeight() const { return m_height; }
	unsigned int getId() const { return m_id; }
	const std::string getPath() const { return m_path; }
	static Texture LoadTexture(const std::string path)
	{
		unsigned int textureID;
		glGenTextures(1, &textureID);

		int width, height, nrComponents;
		unsigned char *data = stbi_load(path.c_str(), &width, &height, &nrComponents, 0);
		if (data)
		{
			GLenum format;
			if (nrComponents == 1)
				format = GL_RED;
			else if (nrComponents == 3)
				format = GL_RGB;
			else if (nrComponents == 4)
				format = GL_RGBA;

			glBindTexture(GL_TEXTURE_2D, textureID);
			glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
			glGenerateMipmap(GL_TEXTURE_2D);

			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

			stbi_image_free(data);
		}
		else
		{
			std::cout << "Texture failed to load at path: " << path << std::endl;
			stbi_image_free(data);
		}

		return Texture(textureID, path, width, height);
	}

private:
	unsigned int m_id;
	std::string m_path;
	int m_width;
	int m_height;
	RotateType m_rotateType;
};


#endif