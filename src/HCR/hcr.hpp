#ifndef HCR_HPP
#define HCR_HPP

namespace hcr{
class HCR {
#define HO_LAYER 1
#define IH_LAYER 0
public:
	
	HCR(const std::map<int, std::vector<std::vector<double>>>& training,
				const std::map<int, std::vector<std::vector<double>>>& test,
				const std::vector<int>& training_order,
				const std::vector<int>& test_order);

	void Train(double mod);
	std::pair<int, int> Test();
private:
	int output_size;
	int input_size;
	std::map<int, std::vector<std::vector<double>>> training_data;
	std::map<int, std::vector<std::vector<double>>> test_data;
	std::vector<int> training_order;
	std::vector<int> test_order;

	std::vector<std::vector<std::vector<double>>> weights;

	void InitializeNN();

	double FeedForward(const std::vector<double>& weights,
				const std::vector<double>& inputs);

	double FeedForward(const std::vector<double>& weights,
				const std::vector<int>& inputs);

	void BackProp(int layer, double confidence,
				std::vector<double> errors, std::vector<double> input);

    double Sigmoid(double val);
	std::vector<double> Sigmoid(const std::vector<double>& vals);
    double DSigmoid(double val);

    double NetError(const std::vector<double>& output,
                    const std::vector<int>& e_output);

	void PrintWeights();

	std::vector<double> OutputError(const std::vector<double>& result,
				const std::vector<int>& expected_result);

	std::vector<double> HiddenError(const std::vector<double>& input,
				const std::vector<double>& output_errors);

	std::vector<double> Feed(std::vector<double> data, int layer);

	std::vector<double> SoftMax(std::vector<double> output);

    std::vector<int> Max(std::vector<double> data);
    std::vector<double> Activation(const std::vector<double>& data);
    std::vector<double> ActivationDerivitive(const std::vector<double>& data);
    double Activate(double val);
    double ActivateDerivitive(double val);

    std::vector<int> ClassificationMatrix(int val, int size);
};
}

#endif
