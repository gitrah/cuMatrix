/*
 * CMap.h
 *
 *  Created on: Apr 16, 2014
 *      Author: reid
 */
#pragma once

#include <map>
#include <pthread.h>
#include <mutex>
using std::map;
using std::mutex;
using std::pair;

template<typename K, typename V>
class CMap {
private:
	mutable mutex the_mutex;
public:
	typedef typename map<K, V>::iterator iterator;
	typedef typename map<K, V>::const_iterator const_iterator;
	typedef pair<const K, V> value_type;
	map<K, V> the_map;

	size_t size() const _GLIBCXX_NOEXCEPT;

	void clear() _GLIBCXX_NOEXCEPT;

    void erase(iterator __position);

    size_t erase(const K& __x);

	iterator insert(iterator __position, const value_type& __x);

	iterator begin() _GLIBCXX_NOEXCEPT;

	iterator end() _GLIBCXX_NOEXCEPT;

	iterator find(const K& __x);

	const_iterator find(const K& __x) const;

	void safePrint();

};
