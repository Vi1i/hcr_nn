#include <map>
#include <vector>
#include <iostream>
#include <cmath>
#include <random>
#include <utility>
#include <thread>

#include "hcr.hpp"


hcr::HCR::HCR(const std::map<int, std::vector<std::vector<double>>>& training,
			const std::map<int, std::vector<std::vector<double>>>& test,
			const std::vector<int>& training_order,
			const std::vector<int>& test_order) {
	this->training_data = training;
	this->test_data = test;
	this->training_order = training_order;
	this->test_order = test_order;

	this->InitializeNN();
}

void hcr::HCR::Train() {
	std::map<int, int> pos;
	for(auto const& place : this->training_order) {
		std::vector<double> data(this->training_data[place][pos[place]]);

		// std::vector<double> ih_data(this->Feed(data, IH_LAYER));
		// std::vector<double> ho_data(this->Feed(ih_data, HO_LAYER));

		std::vector<double> temp(this->Feed(data, IH_LAYER));
		std::vector<double> ih_data(this->Activation(temp));
		temp.clear();
		temp = this->Feed(ih_data, HO_LAYER);
		std::vector<double> ho_data(this->Activation(temp));

		std::vector<int> expected_d(this->ClassificationMatrix(place,
					ho_data.size()));


		std::vector<double> output_error(this->OutputError(ho_data,
					expected_d));
		std::vector<double> hidden_error(this->HiddenError(ih_data,
					output_error));
		// std::cout << std::endl;
		// this->PrintWeights();
		// std::cout << std::endl;

		double net_error(this->NetError(ho_data, expected_d));
		std::thread t1(&hcr::HCR::BackProp, this, HO_LAYER, 0.05, output_error,
					ih_data);
		std::thread t2(&hcr::HCR::BackProp, this, IH_LAYER, 0.05, hidden_error,
					data);
		t1.join();
		t2.join();
		// this->BackProp(HO_LAYER, 0.07, output_error, ih_data);
		// this->BackProp(IH_LAYER, 0.07, hidden_error, data);
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
		// std::vector<int> activated(this->Max(ho_data));
		// std::cout << "Activated : ";
		// for(auto const& val : activated) {
		// 	std::cout << val << " ";
		// }
		// std::cout << std::endl;
		// std::vector<int> expected(this->ClassificationMatrix(place,
		// 			ho_data.size()));
		// std::cout << "Expected " << place << ": ";
		// for(auto const& val : expected) {
		// 	std::cout << val << " ";
		// }
		// std::cout << std::endl;
		// std::cout << "Net Error : " << net_error << std::endl;
		// std::cin.get();
	}
}

void hcr::HCR::PrintWeights() {
	for(auto const& layer : this->weights) {
		std::cout << "LAYER" << std::endl;
		for(auto const& node : layer) {
			for(auto const w : node) {
				std::cout << w << "\t";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

std::pair<int, int> hcr::HCR::Test() {
    int total(0);
    int correct(0);
    int incorrect(0);
    double percent_correct(0.0);

	std::map<int, int> pos;
	for(auto const& place : this->test_order) {
        total++;
        correct++;

		std::vector<double> data(this->test_data[place][pos[place]]);

		std::vector<double> temp(this->Feed(data, IH_LAYER));
		std::vector<double> ih_data(this->Activation(temp));
		temp.clear();
		temp = this->Feed(ih_data, HO_LAYER);
		std::vector<double> ho_data(this->Activation(temp));

        std::vector<int> activated(this->Max(ho_data));
		std::vector<int> expected(this->ClassificationMatrix(place,
					ho_data.size()));
        for(auto z = 0; z < activated.size(); z++) {
            if(activated[z] != expected[z]) {
                incorrect++;
                correct--;
                break;
            }
        }
        pos[place]++;
    }

    return std::pair<int, int>(total, correct);
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

	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(
				-1/std::sqrt((double)this->input_size),1/std::sqrt((double)this->input_size));
				//-0.5,0.5);

	//Clear out the current weights, just incase.
	this->weights.clear();


	// Hidden Layer nodes
	std::vector<std::vector<double>> row;
    for(auto y = 0; y < this->input_size; y++) {
        std::vector<double> values;
        for(auto z = 0; z < this->input_size + 1; z++) {
            values.push_back(distribution(generator));
        }
        row.push_back(values);
    }
    this->weights.push_back(row);

    //Clear out the current row to reuse
	row.clear();

	// Output layer nodes
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
	//result = 1.0 / (std::exp(-1.0 * val) + 1.0);
	result = 1.0 / (std::abs(val) + 1.0);
	return result;
}

double hcr::HCR::DSigmoid(double val) {
	double result(0.0);
    result = std::exp(val) / std::pow(std::exp(val) + 1.0, 2.0);
    //result = val * (1 - val);
	return result;
}

std::vector<int> hcr::HCR::ClassificationMatrix(int val, int size) {
    std::vector<int> result;
    for(auto z = 0; z < size; z++) {
        if(z == val) {
            result.push_back(1);
        }else{
            result.push_back (0);
        }
    }

    return result;
}

double hcr::HCR::NetError(const std::vector<double>& output,
			const std::vector<int>& e_output) {
    double error(0.0);
    for(auto z = 0; z < output.size(); z++) {
        error += std::pow(((double)e_output[z] - output[z]), 2.0);
    }

    error = error * 0.5;
    return error;
}

std::vector<int> hcr::HCR::Max(std::vector<double> sigma) {
    std::vector<int> result;
    int pos(0);
    double prev(sigma[pos]);

    for(auto z = 1; z < sigma.size(); z++) {
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

void hcr::HCR::BackProp(int layer, double confidence,
			std::vector<double> errors, std::vector<double> input) {
	for(auto node = 0; node < this->weights[layer].size(); node++) {
		double err(errors[node]);
		double inp(input[node]);

		for(auto edge = 0; edge < this->weights[layer][node].size(); edge++) {
			double w(this->weights[layer][node][edge]); 
			this->weights[layer][node][edge] = w + (confidence * err *
                        ((edge == 0) ? 1.0 : inp));
		}
	}
}


std::vector<double> hcr::HCR::OutputError(const std::vector<double>& result,
				const std::vector<int>& expected_result) {
	std::vector<double> output_error;

	for(auto node = 0; node < this->weights[HO_LAYER].size(); node++) {
		double y(result[node]);
		double t(expected_result[node]);

		double error((t - y) * y * (1 - y));
		output_error.push_back(error);
	}

	return output_error;
}

std::vector<double> hcr::HCR::HiddenError(const std::vector<double>& input,
				const std::vector<double>& output_errors) {
	std::vector<double> result;

	for(auto node = 1; node < this->weights[IH_LAYER].size(); node++) {
		double error(0.0);
		double sum(0.0);
		double h(input[node - 1]);

		for(auto edge = 0; edge < this->weights[HO_LAYER].size(); edge++) {
			double oe(output_errors[edge]);
			double w(this->weights[HO_LAYER][edge][node]);
			sum += w * oe;
		}

		error = (h * (1 - h)) * sum;
		result.push_back(error);
	}

	return result;
}

std::vector<double> hcr::HCR::Feed(std::vector<double> data, int layer) {
	std::vector<double> result;
	for(auto node = 0; node < this->weights[layer].size(); node++) {
		result.push_back(this->FeedForward(this->weights[layer][node], data));
	}

	return result;
}

std::vector<double> hcr::HCR::SoftMax(std::vector<double> output) {
	std::vector<double> result;
	for(auto const& si : output) {
		double sum(0.0);
		double y(0.0);
		for(auto const& sk : output) {
			sum += std::exp(sk);
		}

		y = std::exp(si) / sum;
		result.push_back(y);
	}

	return result;
}

std::vector<double> hcr::HCR::Activation(const std::vector<double>& data) {
	std::vector<double> result;
	for(auto const& val : data) {
		result.push_back(this->Activate(val));
	}
	return result;
}


std::vector<double> hcr::HCR::ActivationDerivitive(const std::vector<double>& data) {
	std::vector<double> result;
	for(auto const& val : data) {
		result.push_back(this->ActivateDerivitive(val));
	}
	return result;
}

double hcr::HCR::Activate(double val) {
	return this->Sigmoid(val);
}

double hcr::HCR::ActivateDerivitive(double val){
	return this->DSigmoid(val);
}