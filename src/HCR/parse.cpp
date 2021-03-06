#include "parse.hpp"

void hcr::Parse::ReadFile(std::string filepath) {
	std::ifstream ifs(filepath);
	std::string line;
	char deliminator = ',';

	if(ifs.is_open()) {
		while(std::getline(ifs, line)) {
			std::vector<std::string> l_data = hcr::Parse::Split(line,
						deliminator);
			std::vector<double> li_data;
			for(auto const& el : l_data) {
				double i;
				hcr::Parse::str2d(i, el.c_str());
				li_data.push_back(i);
			}

			int key = (int)li_data.back();
			li_data.pop_back();
			std::vector<double> norm_data;
            for(auto const& val : li_data) {
                double norm_val(val /*/ (double) 16*/);
                norm_data.push_back(norm_val);
            }
            this->data[key].push_back(norm_data);
			this->order.push_back(key);
		}
	}

	ifs.close();
}

const std::map<int, std::vector<std::vector<double>>>& hcr::Parse::GetData() {
	return this->data;
}

const std::vector<int>& hcr::Parse::GetOrder() {
	return this->order;
}
