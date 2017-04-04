#include "parse.hpp"

void hcr::Parse::ReadFile(std::string filepath) {
	std::ifstream ifs(filepath);
	std::string line;
	char deliminator = ',';

	if(ifs.is_open()) {
		while(std::getline(ifs, line)) {
			std::vector<std::string> l_data = hcr::Parse::Split(line,
						deliminator);
			std::vector<int> li_data;
			for(auto const& el : l_data) {
				int i;
				hcr::Parse::str2int(i, el.c_str(), 10);
				li_data.push_back(i);
			}
			int key = li_data.back();
			li_data.pop_back();
			this->data[key].push_back(li_data);
			this->order.push_back(key);
		}
	}

	ifs.close();
}

const std::map<int, std::vector<std::vector<int>>>& hcr::Parse::GetData() {
	return this->data;
}

const std::vector<int>& hcr::Parse::GetOrder() {
	return this->order;
}
