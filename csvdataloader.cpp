#include "csvdataloader.h"

CsvDataLoader::CsvDataLoader(const string filename) {
	dataFile_.open(filename.c_str());
}

// Returns 0 if non data were inserted
unsigned CsvDataLoader::getInput(vector<double> &input_vals) {
	input_vals.clear();
	string line;
	getline(dataFile_, line);
	std::istringstream ss(line);
	string value;
	ss >> value;
	// skip the comment
	while (value.compare("#") == 0) {
		line.clear();
		getline(dataFile_, line);
		ss.str(line);
		value.clear();
		ss >> value;
	}

	ss.clear();
	ss.str(line);
	value.clear();
	while (getline(ss, value, ',')) {
		input_vals.push_back(atof(value.c_str()));
		value.clear();
	}
	if (input_vals.size() == 0) {/* nothing added*/ }
	return input_vals.size(); // 0 if nothing was added, empty line
}
