#include <cassert>
#include <algorithm>
#ifdef _WIN32
#include <windows.h>
#endif
#include "thread_pool.h"

static int instanceCount = 0;

#ifdef _WIN32

class ThreadNameHelper
{
public:
    ThreadNameHelper()
    {
        mDllHandle = ::LoadLibraryA("KernelBase.dll");

        if (mDllHandle)
        {
            mSetThreadDescriptionProc = reinterpret_cast<SetThreadDescriptionProc>(
                ::GetProcAddress(mDllHandle, "SetThreadDescription"));
        }
    }

    ~ThreadNameHelper()
    {
        if (mDllHandle)
        {
            ::FreeLibrary(mDllHandle);
        }
    }

    void setCurrentThreadName(const std::string& name) const
    {
        if (!mSetThreadDescriptionProc)
        {
            return;
        }

        const std::wstring wideName(name.begin(), name.end());

        mSetThreadDescriptionProc(GetCurrentThread(), wideName.c_str());
    }

private:
    typedef HRESULT (*SetThreadDescriptionProc)(HANDLE, PCWSTR ppszThreadDescription);

private:
    HMODULE                     mDllHandle = nullptr;
    SetThreadDescriptionProc    mSetThreadDescriptionProc = nullptr;
};

#else

class ThreadNameHelper
{
public:
    void setCurrentThreadName(const std::string& name) const {}
};

#endif

ThreadPool::ThreadPool(int numThreads /* std::thread::hardware_concurrency() */) :
    mShouldShutdown(false), mRunning(false), mThreadCount(numThreads), mId(++instanceCount)
{
    if (!mThreadCount)
    {
        mThreadCount = std::thread::hardware_concurrency();
    }

    mThreads.reserve(mThreadCount);
}

ThreadPool::~ThreadPool()
{
    if (mRunning)
    {
        this->RequestStop();
    }
}

ThreadPool::Ptr ThreadPool::CreatePtr(int numThreads /* = std::thread::hardware_concurrency() */)
{
    return std::make_shared<ThreadPool>(numThreads);
}

void ThreadPool::Run()
{
    static ThreadNameHelper threadNameHelper;

    threadNameHelper.setCurrentThreadName("Trio_ThreadPool");

    while (true)
    {
        std::function<void()> workItem = nullptr;

        // Lock the mutex while popping a work item off the queue, but release it
        // before calling it.
        {
            std::unique_lock<std::mutex> lock(mThreadPoolMutex);

            if (mShouldShutdown)
            {
                break;
            }

            mCond.wait(lock, [&] { return !mWorkItems.empty() || mShouldShutdown; });

            if (mShouldShutdown)
            {
                break;
            }

            workItem = mWorkItems.top().item;
            mWorkItems.pop();
        }

        if (workItem != nullptr)
        {
            /*
             * |) ()   \/\/ () /? /<
             */
            try
            {
                workItem();
            }
            catch (const std::exception& e)
            {
                printf("ThreadPool work function threw exception: %s\n", e.what());
            }
            catch (...)
            {
                printf("ThreadPool work function threw unknown exception\n");
            }
        }
    }
}

void ThreadPool::Start()
{
    std::unique_lock<std::mutex> lock(mThreadPoolMutex);

    if (!mRunning)
    {
        mRunning = true;
        mShouldShutdown = false;
        mThreads.clear();

        for (int i = 0; i < mThreadCount; ++i)
        {
            mThreads.emplace_back(&ThreadPool::Run, this);
        }
    }
}

void ThreadPool::RequestStop()
{
    std::unique_lock<std::mutex> lock(this->mThreadPoolMutex);
    mShouldShutdown = true;
    mCond.notify_all();
}

void ThreadPool::Wait()
{
    assert(mRunning);
    mRunning = false;

    for (auto& thread : mThreads)
    {
        if (thread.joinable() == true)
        {
            thread.join();
        }
    }
}

void ThreadPool::Detach()
{
    assert(mRunning);
    mRunning = false;

    for (auto& thread : mThreads)
    {
        thread.detach();
    }
}

void ThreadPool::Clear()
{
    std::unique_lock<std::mutex> lock(this->mThreadPoolMutex);
    mWorkItems = std::priority_queue<QueueItem>();
}

void ThreadPool::QueueWorkItem(int index, std::function<void()> item)
{
    std::unique_lock<std::mutex> lock(this->mThreadPoolMutex);
    mWorkItems.emplace(index, std::move(item));
    mCond.notify_one();
}

std::size_t ThreadPool::QSize()
{
    return this->mWorkItems.size();
}
