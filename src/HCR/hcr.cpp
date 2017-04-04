#include <map>
#include <vector>
#include <iostream>
#include <cmath>
#include <random>

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

void hcr::HCR::SetTrainingData(const std::map<int,
			std::vector<std::vector<int>>>& training,
			const std::vector<int>& training_order){
	this->training_data = training;
	this->training_order = training_order;
	this->train = true;

}

void hcr::HCR::SetTestingData(const std::map<int,
			std::vector<std::vector<int>>>& test,
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
		std::vector<double> ih_data;
		std::vector<double> ho_data;

		for(auto z = 0; z < this->input_size; z++) {
			ih_data.push_back(this->Sigmoid(this->FeedForward(
						this->weights[IH_LAYER][z], data)));
		}

		for(auto z = 0; z < this->output_size; z++) {
			ho_data.push_back(this->Sigmoid(this->FeedForward(
						this->weights[HO_LAYER][z], ih_data)));
		}

		std::cout << "Calculated: ";
		for(auto const& val : ho_data) {
			std::cout << val << " ";
		}
		std::cout << std::endl;
		std::cout << "Activated : ";
		for(auto const& val : ho_data) {
			std::cout << this->Activation(val) << " ";
		}
		std::cout << std::endl;
		std::string expected(this->Classification(place, ho_data.size()));
		std::vector<double> expected_d(this->ClassificationMatrix(place,
					ho_data.size()));
		double net_error(this->NetError(ho_data, expected_d));
		std::cout << "Expected " << place << ": " << expected << std::endl;;
		std::cout << "Net Error : " << net_error << std::endl;

		std::vector<double> output_error;
		for(auto const& node : ho_data) {
			double y(ho_data[node]);
			double t(expected_d[node]);
			double error((t - y) * y * (1 - y));
			output_error.push_back(error);
		}

		std::vector<double> hidden_error;
		for(auto node = 0; node < this->input_size + 1; node++) {
			double error(0.0);
			double sum(0.0);
			double h(1.0);
			if(node != 0) {
				h = ih_data[node - 1];
			}
			for(auto output = 0; output < output_error.size(); output++) {
				double oe(output_error[output]);
				double w(this->weights[HO_LAYER][output][node]);

				sum += w * oe;
			}
			error = (h * (1 - h)) * sum;
			hidden_error.push_back(error);
		}

		this->BackProp(output_error, HO_LAYER, ho_data);
		this->BackProp(hidden_error, IH_LAYER, ih_data);
		this->PrintWeights();
		pos[place]++;
		std::cin.get();
	}
}

void hcr::HCR::PrintWeights() {
	for(auto const& vector : this->weights) {
		for(auto const& layer : vector) {
			for(auto const w : layer) {
				std::cout << w << "\t";
			}
			std::cout << std::endl;
		}
	}
}

void hcr::HCR::Test() {

}


void hcr::HCR::InitializeNN() {
	this->output_size = this->training_data.size();
	this->input_size = 0;
	int key(0);
	int layers(2);
	for(auto const map : this->training_data) {
		key = map.first;
		break;
	}
	this->input_size = this->training_data.at(key).at(0).size();

	std::cout << "Outputs: " << this->output_size << std::endl;
	std::cout << "Inputs: " << this->input_size << std::endl;
	std::cout << "Layers: " << layers << std::endl;

	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(
				-1/std::sqrt(this->input_size),1/std::sqrt(this->input_size));

	// Weights between input and hidden
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

	// Weights between hidden and output
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
	for(auto z = 1; z < weights.size(); z++) {
			result += weights[z] * inputs[z - 1];
	}
	// Add the bias
	result += weights[0];

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

    error = error / (double) 2;
    return error;
}

double hcr::HCR::Activation(double val) {
	return (val > 0.5) ? 1.0 : 0.0;
}

void hcr::HCR::BackProp(std::vector<double> errors, int layer,
			std::vector<double> input) {
	for(auto z = 0; z < this->weights[layer].size(); z++) {
		for(auto y = 0; y < errors.size(); y++) {
			double w(this->weights[layer][z][y]);
			if(y == 0) { //bias
				this->weights[layer][z][y] = w + 0.1 * errors[y];
			}else{
				this->weights[layer][z][y] = w + 0.1 * errors[y] * input[y];
			}
		}
	}
}