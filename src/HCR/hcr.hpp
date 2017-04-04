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
    double Sigmoid(double val);
    double dSigmoid(double val);
    std::string Classification(int val, int size);
    int Classification(const std::vector<double>& output);
    std::vector<double> ClassificationMatrix(int val, int size);
    double NetError(const std::vector<double>& output,
                    const std::vector<double>& e_output);
	void PrintWeights();
};
}

#endif
