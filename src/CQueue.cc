#include <mutex>
#include <condition_variable>
#include <queue>

template<typename Data>
class concurrent_queue
{
private:
    std::queue<Data> the_queue;
    mutable std::mutex  the_mutex;
public:
    void push(const Data& data)
    {
    	std::unique_lock<std::mutex> lock(the_mutex);
        the_queue.push(data);
    }

    bool empty() const
    {
    	std::unique_lock<std::mutex> lock(the_mutex);
        return the_queue.empty();
    }

    Data& front()
    {
    	std::unique_lock<std::mutex> lock(the_mutex);
        return the_queue.front();
    }

    Data const& front() const
    {
    	std::unique_lock<std::mutex> lock(the_mutex);
        return the_queue.front();
    }

    void pop()
    {
    	std::unique_lock<std::mutex> lock(the_mutex);
        the_queue.pop();
    }
};
