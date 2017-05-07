#include <iostream>
#include <fstream>

#include "net.h"
#include "datasetmanager.h"

namespace {
	const unsigned kFirstInputCol = 0; // column count sta
	const unsigned kNumFold = 5; // 1 = all data training
}

using namespace std;


//++++++++++++++++++ utility Fun +++++++++++++++++++++++++++

// t is the topology in the format num of unit per layer separated by a space
vector<unsigned> getTopology(string t) {
	vector<unsigned> topology;
	// get the defined topology, the separator used is space
	stringstream ss(t);
	while (!ss.eof()) {
		unsigned n;
		ss >> n;
		topology.push_back(n);
	}
	return topology;
}

// form the data build the training set and use the remaining p% for the validation set
void split(const vector<vector<double> > data, vector<vector<double> >& training_set, vector<vector<double> >& validation_set, int p)
{
	vector<double> row;
	unsigned size = (unsigned)data.size()*p / 100;
	validation_set.clear();
	for (unsigned row_n = 0; row_n < size; row_n++){
		for (unsigned k = 0; k < data[row_n].size(); k++){
			row.push_back(data[row_n][k]);
		}
		validation_set.push_back(row);
		row.clear();
	}
	training_set.clear();
	for (unsigned row_n = size; row_n < data.size(); row_n++){
		for (unsigned k = 0; k < data[row_n].size(); k++){
			row.push_back(data[row_n][k]);
		}
		training_set.push_back(row);
		row.clear();
	}
}



int main() {
	bool classification = false;
	vector<unsigned> topology = getTopology("5 5 2");
	unsigned first_target_col = topology[0];

	// scale and translate values needed for the denormalization procedure
	double scale[] = { 0.045906, 0.49896415 }, translate[] = { -1.692065, -22.383271 };
	unsigned epoch_limit = 100000;
	double  best_error = 10000;
	// used in case model selection is not done
	double best_eta = 0.1, best_alpha = 0.1, best_lambda = 0.001, best_valid = 1000;

	cout << "Start!" << endl;
	datasetManager* data_set_manager = new datasetManager("input.txt", kFirstInputCol, first_target_col, kNumFold);
	vector<vector<double> > d_train, d_valid;
	data_set_manager->shuffle_data_set();
	// model selection with simple validation set (hold out), bias on the validation
	split(data_set_manager->get_data_set(), d_train, d_valid, 30);

	int n_trials = 3;

	int hidden_unit[] = { 5, 6, 7, 8, 9 };	// num of hidden unit, set the values of search
	
	if (true){ //set true for hidden unit estimation
		double best_err = 1000.0;
		int b_hu = 2;
		for (int hu = 0; hu < 5; hu = hu + 1){
			cout << "hidden neuron " << hidden_unit[hu] << endl;
			topology[1] = hidden_unit[hu];
			Net* net2 = new Net(topology, classification, first_target_col, scale, translate);

			// training and validation of a given model
			double t_err = 0;
			double valid_error = 0;
			for (int i = 0; i < n_trials; i++){ // 3 trials on the same data, for local minima

				// model building
				t_err += net2->train(d_train, best_eta, best_alpha, best_lambda, epoch_limit, false /*printout*/, false/*mee*/);

				// model estimation
				valid_error += net2->test(d_valid, false/*print*/, false/*mee*/); //true for mee, false for mse
			}
			t_err /= n_trials;
			std::cout << " mean train error: " << t_err;
			valid_error /= n_trials;
			std::cout << " mean valid error: " << valid_error << std::endl;
			if (valid_error < best_err*0.99){ // min increment to increase complexity
				best_err = valid_error;
				b_hu = topology[1];
				cout << "best " << topology[1] << endl;
			}
			std::cout << " best valid error: " << best_err << std::endl;
			delete net2;
		}
		cout << "end hidden unit selection " << b_hu << endl;
		topology[1] = b_hu;
	}
	// build the according net
	Net* net1 = new Net(topology, classification, first_target_col, scale, translate);

	double lambda_values[] = { 0.0, 0.1, 0.01, 0.001 };	// weight decay
	double alpha_values[] = { 0.0, 0.1, 0.5, 0.7 };	// momentum
	double eta_values[] = { 0.001, 0.01, 0.1, 0.5 };	// learning rate

	// (hyper-param tuning)
	// search the best num of hidden unit
	// unsigned best_hidden_unit_numb = 0;
	//model selection with simple validation (hold out)
	if (true){
		double lambda, alpha, eta;
		double t_err = 0;
		double valid_error = 0;
		for (int l = 0; l < 4; l++){
			lambda = lambda_values[l];
			for (int a = 0; a < 4; a++){
				alpha = alpha_values[a];
				for (int e = 0; e < 4; e++){
					eta = eta_values[e];

					std::cout << "eta " << eta;
					std::cout << " alpha " << alpha;
					std::cout << " lambda " << lambda << std::endl;

					// training and validation of a given model
					t_err = 0;
					valid_error = 0;
					for (int i = 0; i < n_trials; i++){ // 3 trials on the same data, for local minima

						// model building
						t_err += net1->train(d_train, eta, alpha, lambda, epoch_limit, false /*printout*/, false/*mee*/);

						// model estimation
						valid_error += net1->test(d_valid, false/*print*/, false/*mee*/); //true for mee, false for mse
					}
					t_err /= n_trials;
					std::cout << " mean train error: " << t_err;
					valid_error /= n_trials;
					std::cout << " mean valid error: " << valid_error << std::endl;

					// save the best params
					if (valid_error < best_valid){
						best_eta = eta; best_alpha = alpha; best_lambda = lambda;
						best_valid = valid_error;
					}

				}
			}
		}
	}
	cout << "best eta " << best_eta << " best alpha " << best_alpha << " best lambda " << best_lambda << endl;

	// MODEL ERROR ESTIMATION via cross validation
	if (true){
		std::cout << "test error estimate... " << endl;
		double test_error = 0;
		while (data_set_manager->hasNextFold()){
			data_set_manager->nextFold();
			net1->train(data_set_manager->get_training_set(), best_eta, best_alpha, best_lambda, epoch_limit, false /*printout*/, true/*mee*/);
			double e = net1->test(data_set_manager->get_validation_set(), false, true); //true for mee, false for mse
			cout << "fold " << data_set_manager->get_actual_fold_num() << " loss(mee) " << e << endl;
			test_error += e;
		}
		test_error /= data_set_manager->getNumOfFolds();
		std::cout << "test error estimate (mee) " << test_error << endl;
	}

	// TRAIN the best model on all the dataset
	net1->train(data_set_manager->get_data_set(), best_eta, best_alpha, best_lambda, epoch_limit, true /*printout*/, true/*mee*/);

	// final TEST on the choosen model
	datasetManager* test_set_manager = new datasetManager("test.txt", kFirstInputCol, first_target_col, kNumFold);
	vector<vector<double> >& d_test = test_set_manager->get_data_set();

	//test_error = net1->test(d_test, true/*print*/, false/*mee*/); //true for mee, false for mse
	//std::cout << "test_error: " << test_error << endl;

	net1->blindTest(d_test);
	std::cout << "Finish!" << endl;

	delete data_set_manager;
	delete net1;

	std::cout << "Press enter to exit";
	std::cout << "\a";
	std::cin.get();
	return 0;
}