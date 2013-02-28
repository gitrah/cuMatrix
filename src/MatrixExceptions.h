/*
 * MatrixExceptions.h
 *
 *  Created on: Dec 3, 2012
 *      Author: reid
 */

#ifndef MATRIXEXCEPTIONS_H_
#define MATRIXEXCEPTIONS_H_

class MatrixException {};

class outOfBounds : MatrixException{};
class columnOutOfBounds : outOfBounds{};
class rowOutOfBounds: outOfBounds{};

class notVector : MatrixException{};
class notRowVector : notVector{};
class notColumnVector : notVector{};

class notSynced: MatrixException{}; // required buffer (host or dev) missing or out of date
class notSyncedDevice: notSynced{}; // required buffer (host or dev) missing or out of date
class notSyncedHost: notSynced{}; // required buffer (host or dev) missing or out of date

class notSquare : MatrixException{};

class badDimensions: MatrixException{};
class matricesOfIncompatibleShape : badDimensions{};
class rowDimsDontAgree: matricesOfIncompatibleShape{};
class columnDimsDontAgree: matricesOfIncompatibleShape{};
class exceedsMaxBlockDim: badDimensions{};

class notImplemented : MatrixException{};
class nNeqPnotImplemented : notImplemented{};
class singularMatrix: MatrixException{};

class memoryException : MatrixException{};
class noDeviceBuffer : memoryException{};
class noHostBuffer : memoryException{};
class alreadyPointingDevice : memoryException{};
class hostReallocation: memoryException{};
class notEnoughSmem: memoryException{};

class TimerException : MatrixException{};
class TimerNotStarted : TimerException{};
class TimerAlreadyStarted : TimerException{};

#endif /* MATRIXEXCEPTIONS_H_ */
