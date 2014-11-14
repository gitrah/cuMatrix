/*
 * CMap.h
 *
 *  Created on: Apr 16, 2014
 *      Author: reid
 */

#ifndef CMAP_H_
#define CMAP_H_
#include <map>
#include <pthread.h>
#include <mutex>

template<typename K, typename V>
class CMap {
private:
	mutable std::mutex the_mutex;
public:
	typedef typename std::map<K, V>::iterator iterator;
	typedef typename std::map<K, V>::const_iterator const_iterator;
	typedef typename std::pair<const K, V> value_type;
	std::map<K, V> the_map;

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

#endif /* CMAP_H_ */
