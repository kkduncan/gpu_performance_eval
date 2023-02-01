#pragma once
#ifndef __THREAD_POOL_H__
#define __THREAD_POOL_H__

#include <vector>
#include <thread>
#include <mutex>
#include <set>
#include <condition_variable>
#include <queue>
#include <functional>
#include <atomic>

/**
 * \brief Thread pooling class
 */
class ThreadPool
{
public:
    // Convenience aliases
    using Ptr = std::shared_ptr<ThreadPool>;
    using Set = std::set<Ptr>;
    
    /**
     * \brief Constructs a thread pool with the specified number of threads
     * \param[in] numThreads The number of threads in the pool
     */
    ThreadPool(int numThreads = std::thread::hardware_concurrency());

    /**
     * \brief Destructor
     */
    virtual ~ThreadPool();

    /**
     * \brief Construct a pointer to a thread pool
     */
    static Ptr CreatePtr(int numThreads = std::thread::hardware_concurrency());

    /**
     * Prevent copying (movable if necessary)
     */
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator= (const ThreadPool&) = delete;
    
    /**
     * \brief Adds a work item to the queue
     * \param[in] index The timestamp of the work item
     * \param[in] item The work to be carried out
     */
    void QueueWorkItem(int index, std::function<void()> item);
        
    /**
     * \brief Starts processing items
     */
    void Start();

    /**
     * \brief Signifies to all running threads that they should stop processing items
     */
    void RequestStop();

    /**
     * \brief Waits for all running threads to complete their operations then cleans up the pool
     */
    void Wait();

    /**
     * \brief Detach all running threads without waiting for their operations to complete
     */
    void Detach();

    /**
     * \brief Clears the work items on the queue
     */
    void Clear();

    /**
     * \brief Get number of items on the queue
     */
    std::size_t QSize();

private:
    /**
     * \brief The work item
     */
    struct QueueItem
    {
        int index = 0;
        std::function<void()> item;

        QueueItem() = default;

        QueueItem(int idx, std::function<void()> item) :
            index(idx), item(std::move(item))
        {}

        // Used to determine priority; items with earlier timestamps have higher priority.
        bool operator<(const QueueItem& other) const
        {
            return index > other.index;
        }
    };

private:
    /**
     * \brief Method executed by each thread in the pool
     */
    void Run();

private:
    volatile std::atomic_bool mShouldShutdown;
    std::atomic_bool mRunning;
    int mThreadCount;
    int mId;
    
    std::priority_queue<QueueItem> mWorkItems;
    std::vector<std::thread> mThreads;
    std::mutex mThreadPoolMutex;
    std::condition_variable mCond;
};

#endif /* __THREAD_POOL_H__ */