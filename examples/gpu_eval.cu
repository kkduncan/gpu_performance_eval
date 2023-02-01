#include <iostream>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <memory>
#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/detail/config.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include "include/thread_pool.h"

struct Add2
{
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        // C[i] = A[i] + B[i]
        thrust::get<2>(t) = thrust::get<0>(t) + thrust::get<1>(t);
    }
};

struct Subtract2
{
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        // C[i] = A[i] - B[i]
        thrust::get<2>(t) = thrust::get<0>(t) - thrust::get<1>(t);
    }
};

struct Multiply2
{
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        // C[i] = A[i] * B[i]
        thrust::get<2>(t) = thrust::get<0>(t) * thrust::get<1>(t);
    }
};

struct Divide2
{
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        // C[i] = A[i] / B[i]
        thrust::get<2>(t) = thrust::get<0>(t) / thrust::get<1>(t);
    }
};

template <typename T>
struct MinMaxPair
{
   T min_val;
   T max_val;
};

template <typename T>
struct MinMaxUnaryOp
  : public thrust::unary_function< T, MinMaxPair<T> >
{
   __host__ __device__
   MinMaxPair<T> operator()(const T& x) const
   {
       MinMaxPair<T> result;
       result.min_val = x;
       result.max_val = x;
       return result;
   }
};

template <typename T>
struct MinMaxBinaryOp
  : public thrust::binary_function< MinMaxPair<T>, MinMaxPair<T>, MinMaxPair<T> >
{
    __host__ __device__
    MinMaxPair<T> operator()(const MinMaxPair<T>& x, const MinMaxPair<T>& y) const
    {
        MinMaxPair<T> result;
        result.min_val = thrust::min(x.min_val, y.min_val);
        result.max_val = thrust::max(x.max_val, y.max_val);
        return result;
    }
};

thrust::host_vector<float> getRandomVector(const size_t N,
                                           unsigned int seed = thrust::default_random_engine::default_seed)
{
    
    thrust::minstd_rand rng(seed);
    thrust::random::normal_distribution<float> dist(128.0f, 32.0f);
    thrust::host_vector<float> temp(N);
    for(size_t i = 0; i < N; i++)
    {
        temp[i] = dist(rng);
    }
    return temp;
}

template <typename T>
struct Square
{
    __host__ __device__
    T operator()(const T& x) const 
    {
        return x * x;
    }
};

/**
 * \brief Main processing pipeline to mimic a forward pass
 */
void processingPipeline(int w, int h)
{
    int N = w * h;
    thrust::device_vector<float> A = getRandomVector(N, 10);
    thrust::device_vector<float> B = getRandomVector(N, 71);
    thrust::device_vector<float> C = getRandomVector(N, 24);
    thrust::device_vector<float> D = getRandomVector(N, 63);

    thrust::device_vector<float> AB(N);
    thrust::device_vector<float> CD(N);

    thrust::device_vector<float> AplusB(N);
    thrust::device_vector<float> CplusD(N);

    thrust::device_vector<float> F(N);
    thrust::device_vector<float> G(N);

    thrust::device_vector<float> H(N);

    MinMaxUnaryOp<float>  unaryOp;
    MinMaxBinaryOp<float> binaryOp;
    Square<float>         squareOp;
    thrust::plus<float>   plusOp;

    // A * B
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin(), AB.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(A.end(), B.end(), AB.end())),
                     Multiply2());

    // C * D
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(C.begin(), D.begin(), CD.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(C.end(), D.end(), CD.end())),
                     Multiply2());

    // A + B
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin(), AplusB.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(A.end(), B.end(), AplusB.end())),
                     Add2());

    // C + D
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(C.begin(), D.begin(), CplusD.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(C.end(), D.end(), CplusD.end())),
                     Add2());

    // 2 * (A + B)
    thrust::transform(AplusB.begin(), AplusB.end(),
                      thrust::constant_iterator<float>(2.f),
                      AplusB.begin(),
                      thrust::multiplies<float>());
    
    // 2 * (C + D)
    thrust::transform(CplusD.begin(), CplusD.end(),
                      thrust::constant_iterator<float>(2.f),
                      CplusD.begin(),
                      thrust::multiplies<float>());

    // F = (A * B) / (2 * (A + B)
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(AB.begin(), AplusB.begin(), F.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(AB.end(), AplusB.end(), F.end())),
                     Divide2());

    // G = (C * D) / (2 * (C + D)
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(CD.begin(), CplusD.begin(), G.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(CD.end(), CplusD.end(), G.end())),
                     Divide2());

    // MinMax(F)
    MinMaxPair<float> minmaxF = thrust::transform_reduce(F.begin(), F.end(), unaryOp, unaryOp(F[0]), binaryOp);

    // MinMax(G)
    MinMaxPair<float> minmaxG = thrust::transform_reduce(G.begin(), G.end(), unaryOp, unaryOp(G[0]), binaryOp);

    // H = F - G
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(F.begin(), G.begin(), H.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(F.end(), G.end(), H.end())),
                     Subtract2());

    // Norm(H)
    float norm = std::sqrt(thrust::transform_reduce(H.begin(), H.end(), squareOp, 0, plusOp));
}


/**
 * @brief Driver
 */
int main(int argc, char *argv[])
{
    int numRuns = 1;
    if (argc == 2)
    {
        numRuns = std::stoi(argv[1]);
    }

    ThreadPool threadPool(8);
    auto start = std::chrono::steady_clock::now();

    if (numRuns > 1)
    {
        threadPool.Start();
        for (int i = 0; i < numRuns; ++i)
        {
            threadPool.QueueWorkItem(i + 1, []() { processingPipeline(640, 384); });
        }

        while(threadPool.QSize() > 0)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
        }

        threadPool.RequestStop();
        threadPool.Wait();
        threadPool.Clear();
    }
    else
    {
        processingPipeline(640, 384);
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    auto processingTime = static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    std::cout << "Total Processing Time: " << processingTime << " ms." << std::endl;
    std::cout << "Time Per Thread: " << processingTime / numRuns << " ms." << std::endl;

    return 0;
}