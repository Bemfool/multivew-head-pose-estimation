#ifndef FILE_UTILS_H
#define FILE_UTILS_H

#include <string>

namespace file_utils
{
	std::string Id2Str(unsigned int id)
	{
		if (id < 10)
			return "0000000" + std::to_string(id);
		else
			return "000000" + std::to_string(id);
	}

	unsigned int Str2Id(std::string str)
	{
		return std::stoi(str);
	}

};

#endif // FILE_UTILS_H