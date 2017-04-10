#ifndef PARSE_HPP
#define PARSE_HPP

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <map>
#include <climits>

namespace hcr {
class Parse {
public:
	/**
	 * Constructor
	 * Initializes the variables used in the various calls
	 */
	Parse() {};

	/**
	 * Deconstructor
	 * Cleans up the class
	 */
	virtual ~Parse() {};

	/**
	 * Getter for data
	 * This is the function to return the data retrieved from reading the file
	 * @return const std::map<int, std::vector<std::vector<int>>>& The data read
	 */
	 const std::map<int, std::vector<std::vector<double>>>& GetData();

	/**
	 * Getter for data order
	 * This is the function to return the order of the data retrieved from
	 * reading the file 
	 * @return const std::vector<int>& The order of the data read
	 */
	 const std::vector<int>& GetOrder();

	/**
	 * Will read a file with 
	 *
	 * @param filepath The path to the file to read
	 */
	void ReadFile(std::string filepath);

    //! str2int() error codes.
    /*! These set of ENUMs are to allow the str2int() to give high detailed
    	errors. */
	enum STR2INT_ERROR {
				SUCCESS,		/*!< When no errors occur while
									 calculating the integer from the
									 string. */  
				OVERFLOW,		/*!< When the value is larger than the
				 					 integer size for the machine. */  
				UNDERFLOW,		/*!< When the integer is a number of
				 					 smaller absolute value than the
				 					 computer can actually store in
				 					 memory. */
				INCONVERTIBLE	/*!< When the string value cannot be 
									 converted into an integer. */  
				};

	/**
	 * This takes a string input and get an integer value out of it, while
	 * paying attention to the errors that could occur.
	 *
	 * @param i Where the integer value will be stored
	 * @param s The string to be converted
	 * @param base the numerical base to convert to.
	 * @return STR2INT_ERROR This details the events of the split
	 */
	static STR2INT_ERROR str2int (int &i, char const *s, int base) {
	    char *end;
	    long  l;
	    errno = 0;
	    l = strtol(s, &end, base);
	    if((errno == ERANGE && l == LONG_MAX) || l > INT_MAX) {
	        return OVERFLOW;
	    }
	    if((errno == ERANGE && l == LONG_MIN) || l < INT_MIN) {
	        return UNDERFLOW;
	    }
	    if(*s == '\0' || *end != '\0') {
	        return INCONVERTIBLE;
	    }
	    i = l;
	    return SUCCESS;
	}

	/**
	 * This templated split method will take a string and split it upon the
	 * deliminator, and return the value.
	 *
	 * @param s The string to be split
	 * @param delim The deliminator to split the string on
	 * @param result Where the split string will be sent
	 */
	template<typename Out>
	static void Split(const std::string &s, char delim, Out result) {
    	std::stringstream ss;
	    ss.str(s);
	    std::string item;
	    while (std::getline(ss, item, delim)) {
	        *(result++) = item;
	    }
	}

	/**
	 * Helper method of Split(const std::string &s, char delim, Out result)
	 *
	 * @param s The string to split
	 * @param delim The deliminator that the string will be split on
	 * @return vector of the split elements from the string
	 */
	static std::vector<std::string> Split(const std::string &s, char delim) {
	    std::vector<std::string> elems;
	    hcr::Parse::Split(s, delim, std::back_inserter(elems));
	    return elems;
	}

private:
	std::map<int, std::vector<std::vector<double>>> data;
	std::vector<int> order;
};
}
#endif
