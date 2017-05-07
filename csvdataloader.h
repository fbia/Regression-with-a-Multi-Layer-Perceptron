#ifndef CSVDATALOADER_H_
#define CSVDATALOADER_H_

#include <vector>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

//***********************************  class CsvDataLoader  *************************
// class to read data from a csv file
class CsvDataLoader {
public:
	CsvDataLoader(const string filename);
	~CsvDataLoader() { dataFile_.close(); }
	bool isEof(void) { return dataFile_.eof(); }
	// Returns the number of input values read from the file, read a full row
	unsigned getInput(vector<double> &input_vals);

private:
	ifstream dataFile_;
};

#endif /* CSVDATALOADER_H_ */
