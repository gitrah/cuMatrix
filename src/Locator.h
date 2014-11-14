/*
 * Locator.h
 *
 *  Created on: Apr 19, 2013
 *      Author: reid
 */

#ifndef LOCATOR_H_
#define LOCATOR_H_

// Now what is this supposed to do again?

template <typename T> class Locator {
private:
	int device;
public:
	Locator(int _device) : device(_device) {}

	inline Locator& locate(CuMatrix<T>& mat) {
		mat.getMgr().locate(device, mat);
		return *this;
	}
};


#endif /* LOCATOR_H_ */
