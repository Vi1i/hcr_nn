#include <map>
#include <vector>
#include <iostream>
#include <cmath>

#include "hcr.hpp"

hcr::HCR::HCR(const std::map<int, std::vector<std::vector<int>>>& training,
	     const std::vector<int>& training_order) {
	this->training_data = training;
	this->training_order = training_order;
	this->test = false;
	this->train = true;
}

hcr::HCR::HCR(const std::map<int, std::vector<std::vector<int>>>& training,
	const std::map<int, std::vector<std::vector<int>>>& test,
	const std::vector<int>& training_order,
	const std::vector<int>& test_order) {

	this->training_data = training;
	this->test_data = test;
	this->training_order = training_order;
	this->test_order = test_order;
	this->test = true;
	this->train = true;

}

void hcr::HCR::SetTrainingData(const std::map<int, std::vector<std::vector<int>>>& training,
					 const std::vector<int>& training_order){
	this->training_data = training;
	this->training_order = training_order;
	this->train = true;

}

void hcr::HCR::SetTestingData(const std::map<int, std::vector<std::vector<int>>>& test,
					const std::vector<int>& test_order) {

	this->test_data = test;
	this->test_order = test_order;
	this->test = true;

}

void hcr::HCR::Train() {
	this->InitializeNN();

	std::map<int, int> pos;
	for(auto const& place : this->training_order) {
		std::vector<int> data(this->training_data[place][pos[place]]);
		std::vector<double> data2;
		std::vector<double> data3;
		for(auto const& weights : this->weights[0]) {
			data2.push_back(this->Sigmoid(this->FeedForward(weights, data)));
		}
		for(auto const& weights : this->weights[1]) {
			data3.push_back(this->Sigmoid(this->FeedForward(weights, data2)));
		}

        std::cout << "Calculated: ";
        for(auto const& val : data3) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
        std::string expected(this->Classification(place, data3.size()));
		std::vector<double> expected_d(this->ClassificationMatrix(place, data3.size()));
        double error(this->NetError(data3, expected_d));
        std::cout << "Expected " << place << ": " << expected << std::endl;;
        std::cout << "Error     : " << error << std::endl;;
		pos[place]++;
        break;
	}
}

void hcr::HCR::PrintWeights() {
	for(auto const& vector : this->weights) {
		for(auto const& layer : vector) {
			for(auto const w : layer) {
				std::cout << w << " ";
			}
			std::cout << std::endl;
		}
	}
}

void hcr::HCR::Test() {

}


void hcr::HCR::InitializeNN() {
	int output_size(this->training_data.size());
	int input_size(0);
	int key(0);
	int layers(2);
	for(auto const map : this->training_data) {
		key = map.first;
		break;
	}
	input_size = this->training_data.at(key).at(0).size();

	std::cout << "Outputs: " << output_size << std::endl;
	std::cout << "Inputs: " << input_size << std::endl;
	std::cout << "Layers: " << layers << std::endl;


	this->weights.clear();
	for(auto x = 0; x < layers; x++) {				//The layers
        if(x + 1 < layers) {
            std::vector<std::vector<double>> row;
            for(auto y = 0; y < input_size; y++) {	//The nodes on the layer
                std::vector<double> values;
                for(auto z = 0; z < input_size + 1; z++) {	//The weights for each node
                    values.push_back(0);
                }
                row.push_back(values);
            }
            this->weights.push_back(row);
        }else{
            std::vector<std::vector<double>> row;
            for(auto y = 0; y < output_size; y++) {	//The nodes on the layer
                std::vector<double> values;
                for(auto z = 0; z < input_size + 1; z++) {	//The weights for each node
                    values.push_back(0);
                }
                row.push_back(values);
            }
            this->weights.push_back(row);
        }
	}
}

double hcr::HCR::FeedForward(const std::vector<double>& weights,
						 	 const std::vector<double>& inputs) {
	double result(0.0);
	for(auto z = 0; z < weights.size(); z++) {
		if(z == 0) {
			result += weights[z] * (double)1;
		}else {
			result += weights[z] * inputs[z - 1];
		}
	}
	//std::cout << this->weights.size() << std::endl;
	return result;
}

double hcr::HCR::FeedForward(const std::vector<double>& weights,
						 	 const std::vector<int>& inputs) {
	double result(0.0);
	std::vector<double> d_inputs;
	for(auto const& val : inputs) {
		d_inputs.push_back(val);
	}
	result = this->FeedForward(weights, d_inputs);
	return result;
}

double hcr::HCR::Sigmoid(double val) {
	double result(0);
	result = 1 / (std::exp(-1 * val) + 1);
	return result;
}

double hcr::HCR::dSigmoid(double val) {
	double result(0);
	result = std::exp(val) / std::pow(std::exp(val) + 1, 2.0);
	return result;
}

std::vector<double> hcr::HCR::ClassificationMatrix(int val, int size) {
    std::vector<double> result;
    for(auto z = 0; z < size; z++) {
        if(z == val) {
            result.push_back(1);
        }else{
            result.push_back (0);
        }
    }

    return result;
}

std::string hcr::HCR::Classification(int val, int size) {
    std::string result("");
    for(auto z = 0; z < size; z++) {
        if(z == val) {
            result += "1 ";
        }else{
            result += "0 ";
        }
    }

    return result;
}

int hcr::HCR::Classification(const std::vector<double>& output) {
    int pos(-1);
    int ones(0);
    int count(0);
    for(auto const& val : output) {
        if(val == 1){
            ones++;
            pos = count;
        }
        count++;
    }

    return (ones == 1) ? pos : -1;
}

double hcr::HCR::NetError(const std::vector<double>& output,
                          const std::vector<double>& e_output) {
    double error(0.0);
    for(auto z = 0; z < output.size(); z++) {\
        error += std::pow(std::abs(e_output[z] - output[z]), 2.0);
    }

    error = error / (double) 2;
    return error;
}
