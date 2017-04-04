#include <iostream>
#include <string>

#include "Func/func.hpp"
#include "HCR/parse.hpp"
#include "HCR/hcr.hpp"

int main(int argc, char * argv[]) {
	std::string cmd(func::Helper::SplitFilename(argv[0]));

	//Verify arguments
	if(argc != 2) {
		func::Helper::Version(std::cout, cmd);
		std::cerr << "usage: " << cmd << " file" << std::endl;
		exit(EXIT_FAILURE);
	}

	std::string filepath(argv[1]);
	if(!func::Helper::FileExists(filepath)) {
		func::Helper::Version(std::cout, cmd);
		std::cerr << "File does not exist or is not able to be read!"
                  << std::endl;
		exit(EXIT_FAILURE);
	}

	//Start data parsement
	hcr::Parse training = hcr::Parse();
	hcr::Parse testing = hcr::Parse();

	training.ReadFile(filepath);

	hcr::HCR hcr = hcr::HCR(training.GetData(), training.GetOrder());
	hcr.Train();
	exit(EXIT_SUCCESS);
}
