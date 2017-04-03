#ifndef HCR_HPP
#define HCR_HPP

namespace hcr{
class HCR {
public:
	HCR() : test(false), train(false) {};
	HCR(const std::map<int, std::vector<std::vector<int>>>& training,
		const std::vector<int>& training_order);
	
	HCR(const std::map<int, std::vector<std::vector<int>>>& training,
		const std::map<int, std::vector<std::vector<int>>>& test,
		const std::vector<int>& training_order,
		const std::vector<int>& test_order);

	void SetTrainingData(const std::map<int, std::vector<std::vector<int>>>& training,
						 const std::vector<int>& training_order);
	void SetTestingData(const std::map<int, std::vector<std::vector<int>>>& test,
						const std::vector<int>& test_order);
	void Train();
	void Test();
private:
	std::map<int, std::vector<std::vector<int>>> training_data;
	std::map<int, std::vector<std::vector<int>>> test_data;
	std::vector<int> training_order;
	std::vector<int> test_order;

	bool test;
	bool train;

	std::vector<std::vector<std::vector<double>>> weights;

	void InitializeNN();
	double FeedForward(const std::vector<double>& weights,
					   const std::vector<double>& inputs);
	double FeedForward(const std::vector<double>& weights,
					   const std::vector<int>& inputs);
	double Activation(double val);
	void PrintWeights();
};
}

#endif