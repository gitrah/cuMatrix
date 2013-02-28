#include "Matrix.h"
#include "MatrixImpl.h"

template <typename T> bool Matrix<T>::verbose = false;
template <typename T> MemMgr<T>* Matrix<T>::mgr = null;
template <typename T> Matrix<T> Matrix<T>::ZeroMatrix(0,0,false);
template <typename T> long Matrix<T>::Constructed = 0;
template <typename T> long Matrix<T>::Destructed = 0;
template <typename T> long Matrix<T>::HDCopied = 0;
template <typename T> long Matrix<T>::DDCopied = 0;
template <typename T> long Matrix<T>::DHCopied = 0;
template <typename T> long Matrix<T>::HHCopied = 0;
template <typename T> long Matrix<T>::MemHdCopied = 0;
template <typename T> long Matrix<T>::MemDdCopied = 0;
template <typename T> long Matrix<T>::MemDhCopied = 0;
template <typename T> long Matrix<T>::MemHhCopied = 0;

template <typename T> uint Matrix<T>::MaxRowsDisplayed = 10;
template <typename T> uint Matrix<T>::MaxColsDisplayed = 10;

template class Matrix<float>;
template class Matrix<double>;
