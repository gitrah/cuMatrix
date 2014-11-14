#include "CMap.h"
#include <condition_variable>

template<typename K, typename V> size_t CMap<K, V>::size() const
		_GLIBCXX_NOEXCEPT {
	std::unique_lock < std::mutex > lock(the_mutex);
	return the_map.size();
}

template<typename K, typename V> void CMap<K, V>::clear() _GLIBCXX_NOEXCEPT {
	std::unique_lock < std::mutex > lock(the_mutex);
	the_map.clear();
}

template<typename K, typename V> void CMap<K, V>::erase(iterator __position) {
	std::unique_lock < std::mutex > lock(the_mutex);
	the_map.erase(__position);
}

template<typename K, typename V> size_t CMap<K, V>::erase(const K& __x) {
	std::unique_lock < std::mutex > lock(the_mutex);
	return the_map.erase(__x);
}
template size_t CMap<float,char>::erase(const float&);
template size_t CMap<double,char>::erase(const double&);
template size_t CMap<ulong,char>::erase(const ulong&);

template<typename K, typename V> typename std::map<K, V>::iterator CMap<K, V>::insert(typename std::map<K, V>::iterator __position, const typename std::pair<const K, V>& __x) {
	std::unique_lock < std::mutex > lock(the_mutex);
	return the_map.insert(__position, __x);
}
template std::map<float,char>::iterator CMap<float,char>::insert(std::map<float,char>::iterator, const std::pair<const float, char>&);
template std::map<double,char>::iterator CMap<double,char>::insert(std::map<double,char>::iterator, const std::pair<const double, char>&);
template std::map<ulong,char>::iterator CMap<ulong,char>::insert(std::map<ulong,char>::iterator, const std::pair<const ulong, char>&);

template<typename K, typename V> typename std::map<K, V>::iterator CMap<K, V>::begin() _GLIBCXX_NOEXCEPT {
	std::unique_lock < std::mutex > lock(the_mutex);
	return the_map.begin();
}
template std::map<float,char>::iterator CMap<float, char>::begin();
template std::map<double,char>::iterator CMap<double, char>::begin();
template std::map<ulong,char>::iterator CMap<ulong, char>::begin();

template<typename K, typename V> typename std::map<K, V>::iterator CMap<K, V>::end() _GLIBCXX_NOEXCEPT {
	std::unique_lock < std::mutex > lock(the_mutex);
	return the_map.end();
}
template<typename K, typename V> typename std::map<K, V>::iterator CMap<K, V>::find(const K& __x) {
	std::unique_lock < std::mutex > lock(the_mutex);
	return the_map.find(__x);
}
template<typename K, typename V> typename std::map<K, V>::const_iterator CMap<K, V>::find(const K& __x) const {
	std::unique_lock < std::mutex > lock(the_mutex);
	return the_map.find(__x);
}

template<typename K, typename V> void CMap<K, V>::safePrint() {
	std::unique_lock < std::mutex > lock(the_mutex);
	printMap("locked",the_map);
}
