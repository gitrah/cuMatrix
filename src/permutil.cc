#include "util.h"

list<uint> b_util::rangeL( uint2 range,  int step) {
	list<uint> l;
	for (uint i = range.x; i < range.y; i+=step)
		l.push_back(i);
	return l;
}

list<list<uint> >b_util::rangeLL( uint2* ranges, int count, int step) {
	list<list<uint> > ll;
	for (int i =0; i < count; i++) {
		ll.push_back( b_util::rangeL( ranges[i],step));
	}
	return ll;
}

template <typename E>  string b_util::toString(list<E> l) {
	stringstream ss;
	for(typename list<E>::iterator i = l.begin(); i != l.end(); i++ ){
		E v = *i;
		ss << v;
		if(v != l.back()) {
			ss << ", ";
		}
	}
	return ss.str();
}
template string b_util::toString(list<uint> );

template <typename E>  void b_util::print(list<E> l) {
	stringstream ss;
	int len = l.size();
	for(typename list<E>::iterator i = l.begin(); i != l.end(); i++ ){
		E v = *i;
		ss << v;
		if(len > 0) {
			ss << ", ";
		}
		len--;
	}
	outln( ss.str() );
}
template  void b_util::print(list<uint>);

template <typename E>  void b_util::printAll(list<list<E>> listOfLists, list<E> instance) {
	  if (!listOfLists.size()) {
	      print( instance);
	      return;
	  }
	  list<E> currentList = listOfLists.front();
	  listOfLists.pop_front();

	  for(typename list<E>::iterator i = currentList.begin(); i != currentList.end(); i++) {
		  instance.push_back(*i);
	      printAll(listOfLists, instance); //recursively invoking with a "smaller" problem
	      instance.pop_back();
	  }
	  listOfLists.push_front(currentList);
}
template  void b_util::printAll(list<list<uint> >, list<uint>);

template <typename E>  int b_util::countAll(list<list<E>> listOfLists, list<E> instance) {
	  if (!listOfLists.size()) {
	      //print( instance);
	      return 1;
	  }
	  list<E> currentList = listOfLists.front();
	  listOfLists.pop_front();
	  int count = 0;
	  for(typename list<E>::iterator i = currentList.begin(); i != currentList.end(); i++) {
		  instance.push_back(*i);
		  count += countAll(listOfLists, instance); //recursively invoking with a "smaller" problem
	      instance.pop_back();
	  }
	  listOfLists.push_front(currentList);
	  return count;
}
template  int b_util::countAll(list<list<uint> >, list<uint>);

