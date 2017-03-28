#ifndef FUNC_HPP
#define FUNC_HPP

#include <string>
#include <fstream>
#include <sys/stat.h>

#include "hcr_nn_config.hpp"

namespace func {
class Helper {
public:
	/**
	 * This is a just a basic little helper method to print the versioning info.
	 *
	 * @param stream This is the stream it will print out to
	 * @param cmd This is the name of the cmd
	 */
	static void Version(std::ostream& stream, const std::string & cmd) {
	    stream << cmd << " Version " << hcr_nn_VERSION_MAJOR << "."
	    			<< hcr_nn_VERSION_MINOR << "." << hcr_nn_VERSION_PATCH
	    			<< std::endl;
	}

	/**
	 * This splits a file path and returns the end.
	 *
	 * @param str The string of the file path to split
	 * @return std::string The end of the file path
	 */
	static std::string SplitFilename(const std::string& str) {
		std::string result;
		size_t found;
	  	
	  	found = str.find_last_of("/\\");
	  	result = str.substr(found + 1);

	  	return result;
	}

	/**
	 * Checks to see if the file exists, though only that something is there,
	 * not verify that it is a correct type. i.e. not a socket, directory, etc.
	 *
	 * @param filepath The path to check if exists
	 * @return bool True if the path exits; False otherwise.
	 */
	static bool FileExists(const std::string& filepath) {
		struct stat buffer;   
		return (stat(filepath.c_str(), &buffer) == 0); 
	}
};
}
#endif