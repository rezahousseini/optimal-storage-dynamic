#include "STCStepsize.h"
#include <math.h>

STCStepsize::STCStepsize(float alpha0, float _beta, float _a, float _b) : Stepsize(alpha0), beta(_beta), a(_a), b(_b) {
}

void STCStepsize::update(void) {
	iteration = iteration+1;
	alpha = alpha0*(b/iteration+a)/(b/iteration+a+pow(iteration, beta));
}
