#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <thread>
#include <vector>
#include <string>
#include <chrono>

int executeGPUEval(const char* gpuID, const int numInstances)
{
    char cmd[1024];
    sprintf(cmd, "CUDA_VISIBLE_DEVICES=%s ./thrust.example.gpu_eval %d", gpuID, numInstances);
    int retVal = system(reinterpret_cast<char*>(cmd));
    return retVal;
}

int main (int argc, char *argv[])
{
    int numInstances = 1;
    std::vector<std::thread> threads;
    auto start = std::chrono::steady_clock::now();

    if (argc == 2)
    {
        numInstances = std::stoi(argv[1]);
    }

    if (system(NULL))
    {
        std::vector<const char*> gpuIDs {
            "GPU-01eb1a5f-5500-055d-4e53-f91555478120"
        };

        printf ("> Executing MIG command...\n");
        for (size_t i = 0; i < gpuIDs.size(); ++i)
        {
            threads.emplace_back(executeGPUEval, gpuIDs[i], numInstances);
        }

        for (size_t i = 0; i < gpuIDs.size(); ++i)
        {
            threads[i].join();
        }
    }
    else
    {
        printf("Unable to execute MIG execution commands!\n");
        return EXIT_FAILURE;
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    auto processingTime = static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    std::cout << "> Total Processing Time: " << processingTime << " ms." << std::endl;
    std::cout << "> Time Per Instance: " << processingTime / numInstances << " ms." << std::endl;

    return EXIT_SUCCESS;
}