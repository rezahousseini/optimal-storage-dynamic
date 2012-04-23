#ifndef STEPSIZE_H_
#define STEPSIZE_H_

class Stepsize {
	public:
		Stepsize(float _alpha0);
		virtual ~Stepsize();
		virtual void update(void);
		virtual float get(void);
	protected:
		float alpha;
		int iteration;
		float alpha0;
};

#endif /* STEPSIZE_H_ */
