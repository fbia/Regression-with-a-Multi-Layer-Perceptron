#include "neuron.h"
#include <iostream>

Neuron::Neuron(unsigned num_outputs, unsigned my_index) {
	for (unsigned c = 0; c < num_outputs; ++c) {
		output_weights_.push_back(Connection());
		output_weights_.back().weight = randomWeight();
	}

	my_index_ = my_index;
	gradient_ = 0;
	//full_gradient_ = 0;
	output_val_ = 0;
}

double Neuron::randomWeight(void) {
	// between[-0.5;0.5]
	return (rand() / double(RAND_MAX)) - 0.5;
}

// set the output of the hidden unit computing output=transferFun(net)
void Neuron::feedForward(const vector<Neuron> &prev_layer) {
	double net = 0.0;

	// Sum the previous layer's outputs (which are our inputs)
	// Include the bias node from the previous layer.
	for (unsigned n = 0; n < prev_layer.size(); ++n) {
		net += prev_layer[n].get_output_val() // X_jn
			* prev_layer[n].output_weights_[my_index_].weight; //W_jn
	}

	output_val_ = Neuron::sigmoid(net);
	//cout << output_val_ <<endl;
}

// compute the neuron output value
// given the prev layer of neuron with the weights to the actual neuron
// true for classification, sigmoidal output unit
// false for regression, linear output unit
void Neuron::feedForwardOutput(const vector<Neuron> &prev_layer, bool classification) {
	double net = 0.0;

	// Sum the previous layer's outputs (which are our inputs)
	// Include the bias node from the previous layer.
	for (unsigned n = 0; n < prev_layer.size(); ++n) {
		net += prev_layer[n].get_output_val() // X_jn
			* prev_layer[n].output_weights_[my_index_].weight; //W_jn
	}
	if (classification)		output_val_ = Neuron::sigmoid(net);
	else		output_val_ = net; // linear
	//cout << output_val_ << endl;

}


void Neuron::updateInputWeights(vector<Neuron> &prev_layer, double eta, double alpha, double lambda) {
	// The weights to be updated are in the Connection container
	// in the neurons in the preceding layer
	for (unsigned n = 0; n < prev_layer.size(); ++n) {
		// the neuron which contain the weights to me
		Neuron &input_neuron = prev_layer[n];
		double oldDeltaWeight = input_neuron.output_weights_[my_index_].delta_weight;

		double newDeltaWeight =
			// Individual input, magnified by the gradient and train rate:
			eta * gradient_ * input_neuron.get_output_val()
			// add momentum = a fraction (alpha) of the previous delta weight
			+ alpha * oldDeltaWeight;

		// weight decay, if lambda != 0 ; lambda already divided by N
		input_neuron.output_weights_[my_index_].delta_weight *= 1.0 - (2.0 * eta*lambda);

		input_neuron.output_weights_[my_index_].delta_weight = newDeltaWeight;
		input_neuron.output_weights_[my_index_].weight += newDeltaWeight;
		//cout << input_neuron.output_weights_[my_index_].weight << endl;
	}
}


void Neuron::compHiddenGradients(const vector<Neuron> &next_layer) {

	// Sum our contributions of the errors at the nodes we feed.
	// do not count the bias neuron in the next layer
	double sum = 0.0;
	for (unsigned k = 0; k < next_layer.size() - 1; k++) {
		sum += output_weights_[k].weight * next_layer[k].gradient_; // sum_k[ W_kj * delta_k ]
	}

	gradient_ = sigmoidDerivative(output_val_) * sum;
	//full_gradient_ += gradient_ ;
}

void Neuron::compOutputGradients(double target_val, bool classification) {
	double delta = target_val - output_val_;
	if (classification)
		gradient_ = delta * sigmoidDerivative(output_val_);
	else
		gradient_ = delta * 1.0; // 1 = linear derivative

	//full_gradient_ += gradient_;
}


// return sum_j( wji^2 )
double Neuron::sumSquaredWeights(){
	double sum = 0.0;
	double w;
	// Sum the weights^2 to the nodes we feed.
	for (unsigned n = 0; n < output_weights_.size(); ++n) {
		w = output_weights_[n].weight;
		sum += w*w;
	}
	return sum;
}


double Neuron::sumWeights(){
	double sum = 0.0;
	double w;
	// Sum the weights^2 to the nodes we feed.
	for (unsigned n = 0; n < output_weights_.size(); ++n) {
		w = output_weights_[n].weight;
		sum += abs(w);
	}
	return sum;
}


// x is the net fun result
double Neuron::sigmoid(double x) {
	if (x < -45) return 0;
	if (x > 45) return 1;
	double res = 1.0 / (1.0 + exp(-x));
	return res;
}


// x is the output of the neuron
double Neuron::sigmoidDerivative(double x) {
	return x * (1.0 - x);
}

void Neuron::reset_output_weights() {
	for (unsigned w = 0; w < output_weights_.size(); ++w) {
		output_weights_[w].weight = randomWeight();
	}
}

Weights Neuron::get_output_weights() {
	Weights weights;
	for (unsigned w = 0; w < output_weights_.size(); ++w) {
		weights.push_back(output_weights_[w].weight);
	}
	return weights;
}

void Neuron::set_output_weights(const Weights weights) {
	for (unsigned w = 0; w < output_weights_.size(); ++w) {
		output_weights_[w].weight = weights[w];
	}
}
