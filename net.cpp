#include <iostream> // per std::cout e std::endl
#include <cassert>  // per assert
#include <cmath>    // per tanh()
#include <cstdlib>  // per EXIT_SUCCESS, e random
#include <algorithm>    // std::random_shuffle
#include <fstream>
#include <ctime>

#include "net.h"
#include "type.h"

// build the net units adding and initializing the bias also
Net::Net(const vector<unsigned> &topology, bool classification, unsigned first_target_col, double *scale, double *translate) {
	classification_ = classification;
	first_target_col_ = first_target_col;
	scale_ = scale;
	translate_ = translate;

	vector<Connection> output_weights_;
	Connection c;
	output_weights_.push_back(c);

	unsigned num_layers = topology.size();
	for (unsigned layer_n = 0; layer_n < num_layers; ++layer_n) {
		layers_.push_back(vector<Neuron>());
		// the output units have no outputs weights
		unsigned num_outputs = layer_n == topology.size() - 1 ? 0 : topology[layer_n + 1];

		// We have a new layer, now fill it with neurons, and
		// add a bias neuron in each layer.
		for (unsigned neuron_n = 0; neuron_n <= topology[layer_n]; ++neuron_n) {
			layers_.back().push_back(Neuron(num_outputs, neuron_n));
		}

		// Force the bias node's output to 1.0 (it was the last neuron pushed in this layer):
		layers_.back().back().set_output_val(1.0);
	}

}


//input the istance x to the network and compute the output Ou of every unit u in the network
void Net::feedForward(const vector<double> &inputVals) {
	assert(inputVals.size() == (layers_[0].size() - 1));
	// Assign the input values into the input neurons, set output = input
	for (unsigned i = 0; i < inputVals.size(); ++i) {
		layers_[0][i].set_output_val(inputVals[i]);
	}
	// forward propagate
	for (unsigned l = 1; l < layers_.size(); ++l) {
		for (unsigned n = 0; n < layers_[l].size() - 1; ++n) {
			if (l != layers_.size() - 1){ // hidden layers 
				layers_[l][n].feedForward(layers_[l - 1]);
			}
			else{
				layers_[l][n].feedForwardOutput(layers_[l - 1], classification_);
			}
		}
	}

}


// Returns the output computed by the outputNeurons
void Net::getOutputs(vector<double> &output_vals) const
{
	output_vals.clear();

	for (unsigned n = 0; n < layers_.back().size() - 1; ++n) {
		output_vals.push_back(layers_.back()[n].get_output_val());
		//cout << "output" << output_vals.back() << endl;
	}
}


// propagate the errors backward through the network
// and update each network weight Wji(i->j) weight of j from i
// lambda already divided by N
void Net::backProp(const vector<double>& target_vals, double eta, double alpha, double lambda) {
	// Calculate overall net error (RMS of output neuron errors) for this pattern
	vector<Neuron> &output_layer = layers_.back();

	// Calculate output layer gradients (delta)
	// do not count the bias neuron
	for (unsigned n = 0; n < output_layer.size() - 1; ++n) {
		output_layer[n].compOutputGradients(target_vals[n], classification_);
	}

	// Calculate hidden layer gradients (delta)
	for (unsigned layerNum = layers_.size() - 2; layerNum > 0; --layerNum) {
		vector<Neuron> &hiddenLayer = layers_[layerNum];
		vector<Neuron> &nextLayer = layers_[layerNum + 1];

		for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
			hiddenLayer[n].compHiddenGradients(nextLayer);
		}
	}

	// For all layers from outputs to first hidden layer,
	// update connection weights
	vector<Neuron> *layer = NULL;
	for (unsigned l = layers_.size() - 1; l > 0; l--) {
		layer = &layers_[l];

		for (unsigned n = 0; n < layer->size() - 1; ++n) {
			(layer->at(n)).updateInputWeights(layers_[l - 1], eta, alpha, lambda);
		}
	}
}


// train one epoch with backprop alg 
// given a training set d_train
// given a learning rate eta
// given a momentum coefficient alpha
// given the first target col
double Net::trainEpoch(vector<vector<double> > d_train, double eta, double alpha, double lambda, bool mee){
	vector<double> input_vals, output_vals, target_vals;
	double epoch_error = 0;
	unsigned train_size = d_train.size();
	unsigned col_size = d_train[0].size(); // fixed number of features

	// randomize the order of the training patterns, using built-in random generator:
	std::srand(unsigned(std::time(0)));
	std::random_shuffle(d_train.begin(), d_train.end());

	// resetFullGradients();

	for (unsigned pattern_i = 0; pattern_i < train_size; pattern_i++) {

		// get the input features of the current pattern
		input_vals.clear();
		for (unsigned feature_j = 0; feature_j < first_target_col_; feature_j++){
			//cout<< "input" << d_train[pattern_i][feature_j] << endl;
			input_vals.push_back(d_train[pattern_i][feature_j]);
		}

		feedForward(input_vals);
		layers_;
		// get the target features of the current pattern
		target_vals.clear();
		for (unsigned feature_j = first_target_col_; feature_j < col_size; feature_j++) {
			target_vals.push_back(d_train[pattern_i][feature_j]);
			//cout << "target" << target_vals.back() << endl;
		}
		getOutputs(output_vals);

		// compute the loss, not related to the backprop computation, just to report
		epoch_error += computeError(target_vals, output_vals, mee); // false for MSE, true for MEE

		backProp(target_vals, eta, alpha, lambda);

	}// end training epoch

	// mean squared error + weight decay with lambda/N
	epoch_error /= train_size;
	return epoch_error;
}


// train the network with backprop alg with a given stopping criterion
// given a training set d_train
// given a learning rate eta
// given a momentum coefficient alpha
// given the first target col
double Net::train(vector<vector<double> > data, double eta, double alpha, double lambda,
	unsigned epoch_max, bool printout, bool mee)
{
	resetNetWeights();
	vector<vector<double> > d_valid; // internal valid set for stopping criterion
	vector<vector<double> > d_train;
	// randomize the order of the training patterns
	std::srand(unsigned(std::time(0)));
	// using built-in random generator:
	std::random_shuffle(data.begin(), data.end());
	split(data, d_train, d_valid, 30); // build the validation set for the early stopping
	ofstream errFile("error.csv");
	// early - stopping parameters
	unsigned patience = 1000;  // look as this many epochs regardless
	int patience_increase = 3000; //   wait this much longer when a new best is found
	double improvement_threshold = 0.999;  // a relative improvement of this much is considered significant
	int validation_frequency = 3; // go through this many n_train_batches before checking the network on the validation set	
	vector <vector<Weights> > best_params; // se voglio i pesi, altrim prendo l'iterazione ma poi rialleno
	double this_validation_loss, best_validation_loss = 100000;
	bool done_looping = false;
	unsigned epoch = 0, b_epoch = 0;
	double t_error;
	while ((epoch < epoch_max) && (!done_looping)){
		// Report "1" for first epoch, "n_epochs" for last epoch
		epoch = epoch + 1;

		t_error = trainEpoch(d_train, eta, alpha, lambda / d_train.size(), mee);

		//note that if we do `iter % validation_frequency` it will be true for iter = 0 
		// which we do not want. We want it true for	iter = validation_frequency - 1.
		if ((epoch % validation_frequency) == 0){

			this_validation_loss = test(d_valid, false/*print*/, mee);	//true for mee, false for mse
			// report actual train and test error
			if (printout) errFile << epoch << ", " << t_error << ", " << this_validation_loss << endl;
			if (this_validation_loss < best_validation_loss){
				//# improve patience if loss improvement is good enough
				//if (this_validation_loss < best_validation_loss * improvement_threshold)
				patience = (patience < epoch + patience_increase) ? epoch + patience_increase : patience;
				// max(patience, epoch + patience_increase)

				best_params = getNetWeights(); // copy the weights or iter numb and retrain
				best_validation_loss = this_validation_loss;
				b_epoch = epoch;
			}
		}
		if (patience <= epoch)
			done_looping = true;
	}

	setNetWeights(best_params);
	d_valid.clear();
	d_train.clear();
	best_params.clear();
	errFile.close();

	cout << " total epochs " << epoch;
	cout << " best epoch " << b_epoch;
	cout << " best train loss(mse) " << best_validation_loss << endl;
	return best_validation_loss;
}


// Compute the Squared Error on a given training sample
// summed over all output units results in the network
double Net::computeError(const vector<double>& target_vals, const vector<double>& output_vals, bool mee) {
	double r = 0.0, delta;
	for (unsigned n = 0; n < target_vals.size(); ++n) {
		delta = target_vals[n] - output_vals[n];
		r += delta * delta;
	}
	if (mee) r = sqrt(r);
	return r;
}


// print into an output file
// <input features, output features, target features, 0>
// every row is closeb by a 0 column
void Net::printResults(ofstream& out_file, const vector<double> input_vals, const vector<double> output_vals, const vector<double> target_vals)const{
	unsigned i;
	for (i = 0; i < input_vals.size(); i++){
		out_file << input_vals[i] << ",";
	}
	for (i = 0; i < output_vals.size(); i++){
		if (classification_){
			if (output_vals[i] < 0.5)
				out_file << "0,";
			else
				out_file << "1,";
		}
		else
			out_file << output_vals[i] << ",";
	}
	for (i = 0; i < target_vals.size(); i++){
		out_file << target_vals[i] << ",";
	}
	out_file << 0 << endl;
}


// compute the error (MSE/MEE) of the net given a test set in the original scale
// if print_output is set true produce the output csv file 
// <input_features, output_features, target_features>
double Net::test(const vector<vector<double> > d_test, bool print_output, bool mee){

	ofstream out_file("output.csv");
	double loss = 0.0;
	double right = 0.0, wrong = 0.0, accuracy = 0.0;
	vector<double> input_vals, output_vals, target_vals;
	unsigned set_size = d_test.size();

	for (unsigned i_sample = 0; i_sample < set_size; i_sample++) {
		input_vals.clear();
		for (unsigned j = 0; j < first_target_col_; j++){
			input_vals.push_back(d_test[i_sample][j]);
		}

		feedForward(input_vals);
		getOutputs(output_vals);
		target_vals.clear();
		unsigned size = d_test[i_sample].size();
		for (unsigned j = first_target_col_; j < size; j++) {
			target_vals.push_back(d_test[i_sample][j]);
		}

		if (classification_){ // check accuracy, 1 if >=0.5; 0 otherwise 
			for (unsigned j = 0; j < target_vals.size(); j++) {
				if (abs(target_vals[j] - output_vals[j]) < 0.5) right = right + 1.0;
				else wrong += 1.0;
			}
		}

		denormalize(output_vals, scale_, translate_);
		denormalize(target_vals, scale_, translate_);

		if (print_output){ // print into an output file
			printResults(out_file, input_vals, output_vals, target_vals);
		}
		loss += computeError(target_vals, output_vals, mee); // false for MSE, true for MEE
		output_vals.clear();

	}// end test

	input_vals.clear(); output_vals.clear(); target_vals.clear();
	out_file.close();

	if (classification_){
		accuracy = right*1.0 / (right + wrong);
		if (print_output) cout << "test accuracy " << accuracy << endl;  // print the accuracy
	}

	loss /= (double)d_test.size(); // mean squared error
	if (classification_)
		return 1.0 - accuracy; // misclassification error
	return loss;

}

// final test for the competition 
// print into the file finalout.csv
// the 2 col of computed outputs
void Net::blindTest(const vector<vector<double> > d_test){
	ofstream finalOut("finalout.csv");
	vector<double> input_vals, output_vals;
	int dim = d_test.size();
	for (int i_sample = 0; i_sample < dim; i_sample++){

		for (unsigned j = 0; j < first_target_col_; j++){
			input_vals.push_back(d_test[i_sample][j]);
		}

		feedForward(input_vals);
		input_vals.clear();

		getOutputs(output_vals);

		denormalize(output_vals, scale_, translate_);

		finalOut << i_sample + 1 << "," << output_vals[0] << "," << output_vals[1] << endl;

		output_vals.clear();
	}
	finalOut.close();
}


// reset the net weigts
void Net::resetNetWeights(void) {
	unsigned num_layers = layers_.size();
	for (unsigned layer = 0; layer < num_layers; ++layer) {
		unsigned num_neurons = layers_[layer].size();
		for (unsigned neuron = 0; neuron < num_neurons; ++neuron) {
			layers_[layer][neuron].reset_output_weights();
		}
	}
}


void Net::setNetWeights(const vector<vector<Weights> > w) {
	unsigned num_layers = layers_.size();
	for (unsigned layer = 0; layer < num_layers; ++layer) {
		unsigned num_neurons = layers_[layer].size();
		for (unsigned neuron_num = 0; neuron_num < num_neurons; ++neuron_num) {
			layers_[layer][neuron_num].set_output_weights(w[layer][neuron_num]);
		}
	}
}


vector<vector<Weights> > Net::getNetWeights() {
	vector<vector<Weights> > weights;
	Weights neuron_weights;
	unsigned num_of_layers = layers_.size();
	weights.resize(num_of_layers);
	unsigned num_neurons;
	for (unsigned layer_index = 0; layer_index < num_of_layers; ++layer_index) {
		num_neurons = layers_[layer_index].size();
		weights[layer_index].resize(num_neurons);
		for (unsigned neuron_n = 0; neuron_n < num_neurons; ++neuron_n) {
			neuron_weights = layers_[layer_index][neuron_n].get_output_weights();
			for (unsigned i = 0; i < neuron_weights.size(); ++i) {
				weights[layer_index][neuron_n].push_back(neuron_weights[i]);
				//cout << " " << neuron_weights[i];
			}
			//cout << endl;
		}
		//cout << endl;
	}
	return weights;
}


// split p% of data for validation and the remaining for test
void Net::split(const vector<vector<double> > data,
	vector<vector<double> >& d_train,
	vector<vector<double> >& d_valid, int p)
{
	vector<double> row;
	unsigned size = (unsigned)data.size()*p / 100;

	d_valid.clear();
	for (unsigned row_n = 0; row_n < size; row_n++){
		for (unsigned k = 0; k < data[row_n].size(); k++){
			row.push_back(data[row_n][k]);
		}
		d_valid.push_back(row);
		row.clear();
	}

	d_train.clear();
	for (unsigned row_n = size; row_n < data.size(); row_n++){
		for (unsigned k = 0; k < data[row_n].size(); k++){
			row.push_back(data[row_n][k]);
		}
		d_train.push_back(row);
		row.clear();
	}
}


// do the sum of the squared weights
// a neuron contains the weights to the neurons of the feeded layer
// the output layer must not be counted
// the bias neuron must not be counted
double Net::sumSquaredWeights(){
	double sum = 0;
	for (unsigned l = 0; l < layers_.size() - 1; ++l) {
		for (unsigned n = 0; n < layers_[l].size() - 1; ++n) {
			sum += layers_[l][n].sumSquaredWeights();
		}
	}
	return sum;
}

double Net::sumWeights(){
	double sum = 0;
	for (unsigned l = 0; l < layers_.size() - 1; ++l) {
		for (unsigned n = 0; n < layers_[l].size() - 1; ++n) {
			sum += layers_[l][n].sumWeights();
		}
	}
	return sum;
}


// data to be denormalized, scaling and translation computed by the normalization
void Net::denormalize(vector<double>& data, double* scale, double* translate){
	for (unsigned i = 0; i < data.size(); i++){
		data[i] = (data[i] - translate[i]) / scale[i];
	}
}

//double Net::sumSquaredGradients(){
//	double sum = 0;
//	for (unsigned l = 0; l < layers_.size() - 1; ++l) {
//		for (unsigned n = 0; n < layers_[l].size() - 1; ++n) {
//			sum += layers_[l][n].squaredGradient();
//		}
//	}
//	return sqrt(sum);
//}




//void Net::resetFullGradients(){
//	for (unsigned l = 0; l < layers_.size() - 1; ++l)
//		for (unsigned n = 0; n < layers_[l].size() - 1; ++n) layers_[l][n].reset_full_gradient();
//}
