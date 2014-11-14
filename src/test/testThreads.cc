#include "../util.h"
#include "tests.h"
#include "../CMap.h"
#include <omp.h>

using namespace std;

template int testCMap<float>::operator()(int argc, char const ** args) const;
template int testCMap<double>::operator()(int argc, char const ** args) const;
template int testCMap<ulong>::operator()(int argc, char const ** args) const;
template <typename T> int testCMap<T>::operator()(int argc, char const ** args) const{

	CMap<T, char> cmap;
	std::map<T,char> smap;

	const char* str = "abcdefghijklmnopqrstuvwxyz";
	uint tid, nthreads;
	// omp info
	outln("outside '#pragma omp parallel' block");


	outln("omp_in_parallel  " << tOrF(omp_in_parallel()));
	outln("omp_get_num_procs " << omp_get_num_procs());
	outln("omp_get_level (nesting level) " << omp_get_level());
	outln("omp_get_nested(able) " << omp_get_nested());
	outln("omp_get_max_active_levels " << omp_get_max_active_levels());
	outln("omp_get_max_threads " << omp_get_max_threads());

	nthreads = omp_get_num_threads();
#pragma omp parallel private(tid)
	{
		tid = omp_get_thread_num();
		if(tid == 0) {
			outln("first thread in omp ||");
			outln("omp_in_parallel  " << tOrF(omp_in_parallel()));
			outln("omp_get_level (nesting level) " << omp_get_level());
			outln("omp_get_nested(able) " << omp_get_nested());
			outln("omp_get_max_active_levels " << omp_get_max_active_levels());
		}
		cmap.insert(cmap.begin(), pair<T,char>((T) tid, str[tid]));
		smap.insert(smap.begin(), pair<T,char>((T) tid, str[tid]));
		#pragma omp final(tid == 11)
		{
			flprintf("last tred tid %d == 11 %s\n", tid, tOrF(tid == 11));
			outln("omp_in_final " << omp_in_final());
		}
	}

	printMap<T, char>("cmap after pop ", cmap.the_map);
	printMap<T, char>("smap after pop ", smap);
#pragma omp parallel private(tid)
	{
		tid = omp_get_thread_num();
		cmap.erase(tid);
		smap.erase(tid);
		if(tid == nthreads/2) {
			printMap<T, char>("cmap during rip ", cmap.the_map);
			printMap<T, char>("smap during rip ", smap);
		}
	}

	printMap<T, char>("cmap after rip ", cmap.the_map);
	printMap<T, char>("smap after rip ", smap);

	return 0;
}
