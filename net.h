#ifndef NET_H_
#define NET_H_
#include <vector>   // per vector

#include "type.h"
#include "neuron.h"

using namespace std;


// ****************** class Net ******************
class Net {
public:
	Net(const vector<unsigned> &topology, bool classification, unsigned first_target_col, double *scale, double *translate);
	~Net() { layers_.clear(); }
	void feedForward(const vector<double> &inputVals); // compute the units output value
	void backProp(const vector<double>& target_vals, double eta, double alpha, double lambda); //  for a single pattern
	double trainEpoch(vector<vector<double> > d_train, double eta, double alpha, double lambda, bool mee); // one epoch of backprop alg throught the training set
	double Net::train(vector<vector<double> > data, double eta, double alpha, double lambda,
		unsigned epoch_max, bool printout, bool mee);
	void getOutputs(vector<double> &output_vals) const;
	void resetNetWeights();
	vector<vector<Weights> > getNetWeights(void);
	void setNetWeights(vector<vector<Weights> > w);
	double computeError(const vector<double>& target_vals, const vector<double>& output_vals, bool mee);
	double Net::test(const vector<vector<double> > d_test, bool print_output, bool mee);
	void printResults(ofstream& out_file, const vector<double> input_vals, const vector<double> output_vals, const vector<double> target_vals)const;
	void Net::blindTest(const vector<vector<double> > d_test);
	double sumWeights();
	void Net::resetFullGradients();

private:
	void Net::denormalize(vector<double>& data, double* scale, double* translate);
	double sumSquaredWeights();
	double sumSquaredGradients();
	void Net::split(const vector<vector<double> > data, vector<vector<double> >& d_train, vector<vector<double> >& d_valid, int p);
	vector<vector<Neuron> > layers_; // m_layers[layerNum][neuronNum]
	bool classification_;
	unsigned first_target_col_;
	double * scale_ = NULL, *translate_ = NULL;
};


#endif /* NET_H_ */
