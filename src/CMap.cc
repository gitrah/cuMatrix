#include "CMap.h"
#include "debug.h"
#include <condition_variable>
#ifdef CuMatrix_UseOmp
#include <omp.h>
#endif
using std::unique_lock;

template<typename T> struct CuMatrix;

template class CMap<float, char> ;
template class CMap<double, char> ;
template class CMap<long, char> ;
template class CMap<ulong, char> ;
template class CMap<float*, string> ;
template class CMap<float const*, long> ;
template class CMap<float const*, float const*> ;
template class CMap<double const*, double const*> ;
template class CMap<long const*, long const*> ;
template class CMap<ulong const*, ulong const*> ;
template class CMap<uint const*, uint const*> ;
template class CMap<int const*, int const*> ;
template class CMap<float*, int> ;
template class CMap<float*, long> ;
template class CMap<double*, string> ;
template class CMap<double*, int> ;
template class CMap<long*, int> ;
template class CMap<ulong*, int> ;
template class CMap<ulong*, long> ;
template class CMap<long*, long> ;
template class CMap<double*, long> ;
template class CMap<long*, string> ;
template class CMap<ulong*, string> ;
template class CMap<int*, string> ;
template class CMap<int*, int> ;
template class CMap<int*, long> ;
template class CMap<uint*, int> ;
template class CMap<uint*, long> ;
template class CMap<uint*, string> ;
template class CMap<string, int> ;
template class CMap<string, long> ;
template class CMap<string, double> ;
template class CMap<CuMatrix<float>*, float*> ;
template class CMap<CuMatrix<double>*, double*> ;
template class CMap<CuMatrix<int>*, int*> ;
template class CMap<CuMatrix<uint>*, uint*> ;
template class CMap<CuMatrix<ulong>*, ulong*> ;
template class CMap<CuMatrix<long>*, long*> ;

template<typename K, typename V> size_t CMap<K, V>::size() const
		_GLIBCXX_NOEXCEPT {
	//tprintf("size in\n");
	unique_lock < mutex > lock(the_mutex);
	return the_map.size();
}

template<typename K, typename V> void CMap<K, V>::clear() _GLIBCXX_NOEXCEPT {
	//tprintf("clear in\n");
	unique_lock < mutex > lock(the_mutex);
	the_map.clear();
}

template<typename K, typename V> void CMap<K, V>::erase(iterator __position) {
	//tprintf("erase(iter) in\n");
	unique_lock < mutex > lock(the_mutex);
	the_map.erase(__position);
}

template<typename K, typename V> size_t CMap<K, V>::erase(const K& __x) {
	//tprintf("erase(key&) in\n");
	unique_lock < mutex > lock(the_mutex);
	return the_map.erase(__x);
/*
	size_t rem = 0;
	the_mutex.lock();
	rem = the_map.erase(__x);
	the_mutex.unlock();
	tprintf("erase(key&) out\n");
	return rem;
*/
}

template<typename K, typename V> typename map<K, V>::iterator CMap<K, V>::insert(typename map<
		K, V>::iterator __position, const pair<const K, V>& __x) {
	//tprintf("insert in\n");
	unique_lock < mutex > lock(the_mutex);
	return the_map.insert(__position, __x);
}

template<typename K, typename V> typename map<K, V>::iterator CMap<K, V>::begin()
		_GLIBCXX_NOEXCEPT {
	//tprintf("begin in\n");
	unique_lock < mutex > lock(the_mutex);
	return the_map.begin();
}

template<typename K, typename V> typename map<K, V>::iterator CMap<K, V>::end()
		_GLIBCXX_NOEXCEPT {
	//tprintf("end in\n");
	unique_lock < mutex > lock(the_mutex);
	return the_map.end();
}

template<typename K, typename V> typename map<K, V>::iterator CMap<K, V>::find(const K& __x) {
	//tprintf("find(key&) in\n");
	unique_lock < mutex > lock(the_mutex);
	return the_map.find(__x);
}

template<typename K, typename V> typename map<K, V>::const_iterator CMap<K, V>::find(const K& __x) const {
	//tprintf("const find(key&) in\n");
	unique_lock < mutex > lock(the_mutex);
	return the_map.find(__x);
}

template<typename K, typename V> void CMap<K, V>::safePrint() {
	//tprintf("safePrint in\n");
	unique_lock < mutex > lock(the_mutex);
	auto it = the_map.begin();
	while(it !=  the_map.end()) {
		cout << "\t" << (*it).first << " -> " << (*it).second << endl;
		it++;
	}
}
