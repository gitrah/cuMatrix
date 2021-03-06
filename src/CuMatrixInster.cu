#include "CuMatrix.h"

template <typename T> MemMgr<T>* CuMatrix<T>::mgr = null;

template <typename T> long CuMatrix<T>::Constructed = 0;
template <typename T> long CuMatrix<T>::Destructed = 0;
template <typename T> long CuMatrix<T>::HDCopied = 0;
template <typename T> long CuMatrix<T>::DDCopied = 0;
template <typename T> long CuMatrix<T>::DHCopied = 0;
template <typename T> long CuMatrix<T>::HHCopied = 0;
template <typename T> long CuMatrix<T>::MemHdCopied = 0;
template <typename T> long CuMatrix<T>::MemDdCopied = 0;
template <typename T> long CuMatrix<T>::MemDhCopied = 0;
template <typename T> long CuMatrix<T>::MemHhCopied = 0;

/*
template <typename T> CMap<string, int>  CuMatrix<T>::methodDhCopies;
template <typename T> CMap<string, long>  CuMatrix<T>::methodDhVolumes;
template <typename T> CMap<string, double>  CuMatrix<T>::methodDhTimes;
*/

template <typename T> const T CuMatrix<T>::MinValue = util<T>::minValue();
template <typename T> const T CuMatrix<T>::MaxValue = util<T>::maxValue();
template <typename T> int CuMatrix<T>::MaxRowsDisplayed = 0;
template <typename T> int CuMatrix<T>::MaxColsDisplayed = 0;
template <typename T> dim3 CuMatrix<T>::DefaultMatProdBlock = dim3(32,32);
template <typename T> CuMatrix<T>* CuMatrix<T>::Identities[1024];

template <typename T> curandState * CuMatrix<T>::devStates= null;
//template <typename T> string CuMatrix<T>::theTypeStr = "";
template <typename T> DMatrix<T> DMatrix<T>::ZeroMatrix(0,0,0,0);
template struct DMatrix<float>;
template struct DMatrix<double>;
template struct DMatrix<int>;
template struct DMatrix<long>;
template struct DMatrix<uint>;
template struct DMatrix<ulong>;


template <typename T> CuMatrix<T> CuMatrix<T>::ZeroMatrix(0,0,false,false);
template class CuMatrix<float>;
template class CuMatrix<double>;
template class CuMatrix<int>;
template class CuMatrix<long>;
template class CuMatrix<uint>;
template class CuMatrix<ulong>;

template class PackedMat<float>;
template class PackedMat<double>;
template class PackedMat<int>;
template class PackedMat<long>;
template class PackedMat<uint>;
template class PackedMat<ulong>;
