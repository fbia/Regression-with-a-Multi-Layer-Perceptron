#ifndef TYPE_H_
#define TYPE_H_

#include <vector>
using namespace std;

struct Connection {
	double weight;
	double delta_weight;
};

typedef vector<double> Weights;


#endif /* TYPE_H_ */
