#ifndef DATASETMANAGER_H_
#define DATASETMANAGER_H_

#include <iostream> // per std::cout e std::endl
#include <vector>   // per vector
#include <ctime>        // std::time
#include <algorithm>    // std::random_shuffle

#include "csvdataloader.h"

using namespace std;


class datasetManager : public CsvDataLoader {
public:
	datasetManager(const string file_name, unsigned input_start_col, unsigned target_start_col, unsigned num_of_fold);
	~datasetManager(){ data_set_.clear(); training_set_.clear(); validation_set_.clear(); }
	void shuffleTrainingset(void);
	void shuffle_data_set(void);
	void nextFold(void);
	bool hasNextFold(void) { return actual_fold_num_ < num_fold_ - 1; }
	int get_actual_fold_num() { return actual_fold_num_; }
	vector<vector<double> > get_training_set() { return training_set_; }
	vector<vector<double> > get_validation_set() { return validation_set_; }
	vector<vector<double> > get_data_set() { return data_set_; }
	unsigned getNumOfFolds(){ return num_fold_; }

private:
	void initKthFold(unsigned k);
	vector<vector<double> > data_set_; // matrix containing the training data input + target
	vector<vector<double> > training_set_; // training set
	vector<vector<double> > validation_set_; // validation set
	unsigned input_start_col_; // col num start by 0
	unsigned target_start_col_; // the col numb of the first target val
	int actual_fold_num_;
	int num_fold_;
};


#endif /* DATASETMANAGER_H_ */
