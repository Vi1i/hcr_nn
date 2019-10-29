#include <iostream>
#include <sstream>
#include <string>

#include "Func/func.hpp"
#include "HCR/parse.hpp"
#include "HCR/hcr.hpp"

int main(int argc, char * argv[]) {
	std::string cmd(func::Helper::SplitFilename(argv[0]));

	//Verify arguments
	if(argc != 3) {
		func::Helper::Version(std::cout, cmd);
		std::cerr << "usage: " << cmd << " training_file testing_file" << std::endl;
		exit(EXIT_FAILURE);
	}

	std::string filepath(argv[1]);
	std::string t_filepath(argv[2]);
	if(!func::Helper::FileExists(filepath)) {
		func::Helper::Version(std::cout, cmd);
		std::cerr << "File does not exist or is not able to be read!"
                  << std::endl;
		exit(EXIT_FAILURE);
	}
	if(!func::Helper::FileExists(t_filepath)) {
		func::Helper::Version(std::cout, cmd);
		std::cerr << "File does not exist or is not able to be read!"
                  << std::endl;
		exit(EXIT_FAILURE);
	}

	//Start data parsement
	hcr::Parse training = hcr::Parse();
	hcr::Parse testing = hcr::Parse();

	training.ReadFile(filepath);
	testing.ReadFile(t_filepath);

	hcr::HCR hcr(training.GetData(), testing.GetData(), training.GetOrder(), testing.GetOrder());

	double epochs(10000);
	//std::string line("0%");
	//std::cout <<"Training: " << line << std::flush;
	std::cout << "EPOCH,TOTAL,CORRECT,ACCURACY(%)" << std::endl;
	for(auto z = 0; z < epochs; z++) {
		hcr.Train(z / 10.0);

		// std::cout << std::string(line.length(), '\b');
		// std::stringstream ss;
		// ss << (z / epochs) * 100  << "%";
		// line = ss.str();
		// std::cout << line << std::flush;
		auto p(hcr.Test());
		std::cout << z + 1 << "," << p.first << "," << p.second << "," << (p.second / (double)p.first) << std::endl;
	}
	//std::cout << std::string(line.length(), '\b') << "100%" << std::endl;

	//hcr.Test();

	exit(EXIT_SUCCESS);
}
