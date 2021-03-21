#ifndef TEXTURE_H
#define TEXTURE_H

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <string>
#include <iostream>

#include "glog/logging.h"

#include "db_params.h"

class Texture
{
public:
	Texture() { }
 	Texture(std::string path, int width, int height)
	  : m_path(path), m_width(width), m_height(height), m_rotateType(RotateType_Invalid) { }

	void setRotateType(RotateType rotateType) { m_rotateType = rotateType; }
	RotateType getRotateType() const { return m_rotateType; }
	int getWidth() const { return m_width; }
	int getHeight() const { return m_height; }
	const std::string getPath() const { return m_path; }
	
	static Texture LoadTexture(const std::string path)
	{
		int width = -1, height = -1, nrComponents;
		unsigned char *data = stbi_load(path.c_str(), &width, &height, &nrComponents, 0);
		if (!data)
		{
			LOG(ERROR) << "Texture failed to load at path: " << path;
		}
		stbi_image_free(data);
		return Texture(path, width, height);
	}

private:
	std::string m_path;
	int m_width;
	int m_height;
	RotateType m_rotateType;
};


#endif