#include "Stepsize.h"

Stepsize::Stepsize(float _alpha0) : alpha(_alpha0), alpha0(_alpha0) {
	iteration = 0;
}

void Stepsize::update(void) {
	iteration = iteration+1;
}

float Stepsize::get(void) {
	return alpha;
}

Stepsize::~Stepsize() {
	// TODO Auto-generated destructor stub
}
