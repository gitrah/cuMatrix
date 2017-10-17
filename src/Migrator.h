/*
 * Migrator.h
 *
 *  Created on: Apr 19, 2013
 *      Author: reid
 */

#pragma once

template <typename T> class Migrator {
private:
	int device;
public:
	Migrator(int _device) : device(_device) {}

	inline Migrator& migrate(CuMatrix<T>& mat) {
		mat.getMgr().migrate(device, mat);
		return *this;
	}
};
