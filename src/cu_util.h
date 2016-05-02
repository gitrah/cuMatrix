/*
 * cuutil.h
 *
 *  Created on: Jul 19, 2014
 *      Author: reid
 */

#pragma once

/*
#include "util.h"

template<typename T> struct DMatrix;
template<typename T> class CuMatrix;


template <typename T> struct cu_util : public util<T> {
	static string pdm(const DMatrix<T>& md);
	static __host__ __device__ void prdm(const DMatrix<T>& md);
	static __host__ __device__ void prdmln(const DMatrix<T>& md);
	static __host__ __device__ void printDm( const DMatrix<T>& dm, const char* msg = null);
	static __host__ __device__ void printRow(const DMatrix<T>& dm, int row = 0);

	static int release(std::map<std::string, CuMatrix<T>*>& map);
	static cudaError_t migrate(int dev, CuMatrix<T>& m, ...);

	static void parseDataLine(string line, T* elementData,
			unsigned int currRow, unsigned int rows, unsigned int cols,
			bool colMajor);
	static void parseCsvDataLine(const CuMatrix<T>* x, int currLine, string line, const char* sepChars);
	static map<string, CuMatrix<T>*> parseOctaveDataFile(
			const char * path, bool colMajor, bool matrixOwnsBuffer = true);
	static map<string, CuMatrix<T>*> parseCsvDataFile(
			const char * path, const char * sepChars, bool colMajor, bool matrixOwnsBuffer = true, bool hasXandY = false);
	static T timeReps( CuMatrix<T> (CuMatrix<T>::*unaryFn)(), const char* name, CuMatrix<T>* mPtr, int reps) ;

};
*/
