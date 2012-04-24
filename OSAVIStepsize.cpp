#include "OSAVIStepsize.h"
#include <math.h>

OSAVIStepsize::OSAVIStepsize(float alpha0, float c0, float _nu, float sigma0, float _gamma) : 
 Stepsize(alpha0), c(c0), nu(_nu), sigma(sigma0), gamma(_gamma), lambda(pow(alpha0, 2)), delta(alpha0) {
}

void OSAVIStepsize::update(void) {
}

void OSAVIStepsize::update(float cost) {
	iteration = iteration+1;
	
	c = (1-nu)*c+nu*cost;
	sigma = (1-nu)*sigma+nu*(c-cost);
	
	alpha = ((1-gamma)*lambda*sigma+pow(1-(1-gamma)*delta, 2)*pow(c, 2))/(pow(1-gamma, 2)*lambda*sigma+pow(1-(1-gamma)*delta, 2)*pow(c, 2)+sigma);
	
	lambda = pow(alpha, 2)+pow(1-(1-gamma)*alpha, 2)*lambda;
	delta = alpha+(1-(1-gamma)*alpha)*delta;
}

