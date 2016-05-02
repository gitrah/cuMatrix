/*
 * MatrixExceptions.h
 *
 *  Created on: Dec 3, 2012
 *      Author: reid
 */

#ifndef MATRIXEXCEPTIONS_H_
#define MATRIXEXCEPTIONS_H_
using std::ostream;

class MatrixException {
public:
	string msg;
	MatrixException() : msg("") {};
	MatrixException(string msg) : msg(msg) {};
	inline friend ostream& operator<<(ostream& os, const MatrixException& e)  {
		return os << e.msg;
	}
};

class illegalArgument: public  MatrixException{};


class outOfBounds : public  MatrixException{};
class columnOutOfBounds : public  outOfBounds{};
class rowOutOfBounds: public  outOfBounds{};

class notVector : public  MatrixException{};
class notRowVector : public  notVector{};
class notColumnVector : public  notVector{};

class notSynced: public  MatrixException{}; // required buffer (host or dev) missing or out of date
class notSyncedDev: public  notSynced{}; // required buffer (host or dev) missing or out of date
class notSyncedHost: public  notSynced{}; // required buffer (host or dev) missing or out of date

class notSquare : public  MatrixException{};

class badDimensions: public  MatrixException{};
class matricesOfIncompatibleShape : public  badDimensions{};
class rowDimsDisagree: public  matricesOfIncompatibleShape{};
class columnDimsDisagree: public  matricesOfIncompatibleShape{};
class exceedsMaxBlockDim: public  badDimensions{};

class notImplemented : public  MatrixException{};
class nNeqPnotImplemented : public  notImplemented{};
class singularMatrix: public  MatrixException{};

class MemoryException : public  MatrixException{};
class noDeviceBuffer : public  MemoryException{};
class noHostBuffer : public  MemoryException{};
class alreadyPointingDevice : public  MemoryException{};
class hostReallocation: public  MemoryException{};
class notEnoughSmem: public  MemoryException{};
class notResidentOnDevice : public  MemoryException{};

class TimerException : public  MatrixException{};
class timerNotStarted : public  TimerException{};
class timerAlreadyStarted : public  TimerException{};

class StreamException: public  MatrixException{};
class wrongStream : public  StreamException{};

class HardwareException : public  MatrixException{};
class insufficientGPUCount : public  MatrixException{};

class PointException : public  MatrixException{};
class TimeNotSet : public PointException {};
class SuperposNotImplemented : public PointException {};
class OglCbException : public MatrixException{};
class ThisAlreadySet : public OglCbException {};

class CapsException : public  MatrixException{};

/*
*/
#endif /* MATRIXEXCEPTIONS_H_ */
