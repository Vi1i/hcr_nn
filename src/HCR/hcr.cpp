#include <map>
#include <vector>
#include <iostream>
#include <cmath>
#include <random>

#include "hcr.hpp"

hcr::HCR::HCR(const std::map<int, std::vector<std::vector<int>>>& training,
	     const std::vector<int>& training_order) {
}

hcr::HCR::HCR(const std::map<int, std::vector<std::vector<double>>>& training,
			const std::map<int, std::vector<std::vector<double>>>& test,
			const std::vector<int>& training_order,
			const std::vector<int>& test_order) {

	this->training_data = training;
	this->test_data = test;
	this->training_order = training_order;
	this->test_order = test_order;
	this->test = true;
	this->train = true;
	this->InitializeNN();
}

void hcr::HCR::SetTrainingData(const std::map<int,
			std::vector<std::vector<int>>>& training,
			const std::vector<int>& training_order){
	//this->InitializeNN();
}

void hcr::HCR::SetTestingData(const std::map<int,
			std::vector<std::vector<int>>>& test,
			const std::vector<int>& test_order) {
	//this->InitializeNN();
}

void hcr::HCR::Train() {
	std::map<int, int> pos;
	for(auto const& place : this->training_order) {
		std::vector<double> data(this->training_data[place][pos[place]]);

		std::vector<double> ih_data(this->Feed(data, IH_LAYER));
		std::vector<double> ho_data(this->Feed(ih_data, HO_LAYER));

        std::vector<double> activated(this->Activation(ho_data));
        std::string expected(this->Classification(place, ho_data.size()));
		std::vector<double> expected_d(this->ClassificationMatrix(place,
					ho_data.size()));
		double net_error(this->NetError(ho_data, expected_d));

		std::vector<double> output_error(this->OutputError(ho_data,
					expected_d));
		std::vector<double> hidden_error(this->HiddenError(ih_data,
					output_error));
		// std::cout << std::endl;
		// this->PrintWeights();
		// std::cout << std::endl;
		this->BackProp(output_error, HO_LAYER, ho_data);
		this->BackProp(hidden_error, IH_LAYER, ih_data);
		pos[place]++;
		// this->PrintWeights();
		// std::cout << "Input Input: ";
		// for(auto const& val : data) {
		// 	std::cout << val << " ";
		// }
		// std::cout << std::endl;
		// std::cout << "Input Hidden: ";
		// for(auto const& val : ih_data) {
		// 	std::cout << val << " ";
		// }
		// std::cout << std::endl;
		// std::cout << ih_data.size() << ":" << ho_data.size() << std::endl;
		// std::cout << "Calculated: ";
		// for(auto const& val : ho_data) {
		// 	std::cout << val << " ";
		// }
		// std::cout << std::endl;
		// std::cout << "Activated : ";
		// for(auto const& val : activated) {
		// 	std::cout << val << " ";
		// }
		// std::cout << std::endl;
		// std::cout << "Expected " << place << ": " << expected << std::endl;;
		// std::cout << "Net Error : " << net_error << std::endl;
		// std::cin.get();
	}
}

void hcr::HCR::PrintWeights() {
	for(auto const& layer : this->weights) {
		for(auto const& node : layer) {
			for(auto const w : node) {
				std::cout << w << "\t";
			}
			std::cout << std::endl;
		}
	}
}

void hcr::HCR::Test() {
    int total(0);
    int correct(0);
    int incorrect(0);
    double percent_correct(0.0);

	std::map<int, int> pos;
	for(auto const& place : this->test_order) {
        ++total;
		std::vector<double> data(this->test_data[place][pos[place]]);

		std::vector<double> ih_data(this->Feed(data, IH_LAYER));
		std::vector<double> ho_data(this->Feed(ih_data, HO_LAYER));

        std::vector<double> activated(this->Activation(ho_data));
		std::vector<double> expected(this->ClassificationMatrix(place,
					ho_data.size()));

        correct++;
        for(auto z = 0; z < activated.size(); z++) {
            if(activated[z] != expected[z]) {
                incorrect++;
                correct--;
                break;
            }
        }
    }

    std::cout << "Tested " << total << " items." << std::endl;
    std::cout << "CORRECT/INCORRECT\t" << correct << "/"
                << incorrect << std::endl;
    std::cout << "Accuracy: " << (correct / (double) total) * 100
                << "%" << std::endl;
}


void hcr::HCR::InitializeNN() {
	this->output_size = this->training_data.size();
	this->input_size = 0;
	int key(0);
	int layers(1);
	for(auto const map : this->training_data) {
		key = map.first;
		break;
	}
	this->input_size = this->training_data.at(key).at(0).size();

	std::cout << "Outputs      : " << this->output_size << std::endl;
	std::cout << "Inputs       : " << this->input_size << std::endl;
	std::cout << "Hidden Layers: " << layers << std::endl;

	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(
				-1/std::sqrt(this->input_size),1/std::sqrt(this->input_size));

	// Hidden Layer nodes
	this->weights.clear();
	std::vector<std::vector<double>> row;
    for(auto y = 0; y < this->input_size; y++) {
        std::vector<double> values;
        for(auto z = 0; z < this->input_size + 1; z++) {
            values.push_back(distribution(generator));
        }
        row.push_back(values);
    }
    this->weights.push_back(row);

	// Output layer nodes
	row.clear();

    for(auto y = 0; y < this->output_size; y++) {
        std::vector<double> values;
        for(auto z = 0; z < this->input_size + 1; z++) {
            values.push_back(distribution(generator));
        }
        row.push_back(values);
    }
    this->weights.push_back(row);
}

double hcr::HCR::FeedForward(const std::vector<double>& weights,
			const std::vector<double>& inputs) {
	double result(0.0);

	// Add the bias
	result += weights[0];

	for(auto z = 1; z < weights.size(); z++) {
			result += weights[z] * inputs[z - 1];
	}

	return result;
}

double hcr::HCR::Sigmoid(double val) {
	double result(0.0);
	result = 1.0 / (1.0 + std::abs(val));
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
    for(auto z = 0; z < output.size(); z++) {
        error += std::pow((e_output[z] - output[z]), 2.0);
    }

    error = error * 0.5;
    return error;
}

std::vector<double> hcr::HCR::Activation(std::vector<double> sigma) {
    std::vector<double> result;
    int pos(0);
    double prev(sigma[pos]);
    for(auto z = 0; z < sigma.size(); z++) {
        if(prev < sigma[z]) {
            prev = sigma[z];
            pos = z;
        }
    }
    for(auto z = 0; z < sigma.size(); z++) {
        if(pos == z) {
            result.push_back(1.0);
        }else{
            result.push_back(0.0);
        }
    }
	return result;
}

void hcr::HCR::BackProp(std::vector<double> errors, int layer,
			std::vector<double> input) {
	double conf(0.5);
	for(auto node = 0; node < this->weights[layer].size(); node++) {
		double err(errors[node]);
		double inp(input[node]);

		for(auto edge = 0; edge < this->weights[layer][node].size(); edge++) {
			double w(this->weights[layer][node][edge]); 

			if(edge == 0) { //bias
				this->weights[layer][node][edge] = w + (conf * err);
			}else{
				this->weights[layer][node][edge] = w + (conf * err * inp);
			}
		}
	}
}


std::vector<double> hcr::HCR::OutputError(const std::vector<double>& result,
				const std::vector<double>& expected_result) {
	std::vector<double> output_error;
	
	//std::cout << std::endl;
	for(auto node = 0; node < this->weights[HO_LAYER].size(); node++) {
		double y(result[node]);
		double t(expected_result[node]);

		double error((t - y) * y * (1.0 - y));
		//std::cout << "(" << t << " - " << y << ") * " << y << " * (1 -" << y << ") = " << error <<std::endl;
		output_error.push_back(error);
	}

	return output_error;
}

std::vector<double> hcr::HCR::HiddenError(const std::vector<double>& input,
				const std::vector<double>& output_errors) {

	std::vector<double> hidden_error;

	for(auto node = 0; node < this->weights[IH_LAYER].size(); node++) {
		double error(0.0);
		double sum(0.0);
		double h((node == 0) ? 1.0 : input[node - 1]);

		for(auto edge = 0; edge < this->weights[HO_LAYER].size(); edge++) {
			double oe(output_errors[edge]);
			double w(this->weights[HO_LAYER][edge][node]);

			sum += w * oe;
		}
		error = (h * (1 - h)) * sum;
		hidden_error.push_back(error);
	}

	return hidden_error;
}

std::vector<double> hcr::HCR::Feed(std::vector<double> data, int layer) {
	std::vector<double> result;
	for(auto node = 0; node < this->weights[layer].size(); node++) {
		result.push_back(this->Sigmoid(this->FeedForward(
					this->weights[layer][node], data)));
	}

	return result;
}