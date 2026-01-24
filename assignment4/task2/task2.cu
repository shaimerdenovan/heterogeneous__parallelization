/*Задание 2
Реализуйте CUDA-программу для вычисления префиксной суммы (сканирования)
массива с использованием разделяемой памяти. Сравните время выполнения с
последовательной реализацией на CPU для массива размером 1 000 000 элементов.*/

/*подключаем CUDA Runtime API*/
#include <cuda_runtime.h>
/*библиотека для вывода в консоль*/
#include <iostream>
/*библиотека vector для хранения данных на CPU*/
#include <vector>
/*библиотека random для генерации случайных чисел*/
#include <random>
/*библиотека chrono для замера времени*/
#include <chrono>

/*макрос для проверки ошибок CUDA*/
#define CUDA_CHECK(call) do {                                   \
    cudaError_t err = call;                                     \
    if (err != cudaSuccess) {                                   \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)  \
                  << " at " << __FILE__ << ":" << __LINE__      \
                  << std::endl;                                 \
        std::exit(1);                                           \
    }                                                           \
} while(0)

/*вспомогательная функция для вычисления ceil(a/b)*/
static inline int ceil_div(int a, int b){ return (a + b - 1) / b; }

/*CUDA-ядро: блочный exclusive scan,алгоритм Blelloch
вычисляется сканирование внутри одного блока с использованием shared памяти
если указатель blockSums не равен nullptr, то в него записывается сумма блока*/
__global__ void scan_block_exclusive(const float* in, float* out, float* blockSums, int n) {
    /*динамическая разделяемая память*/
    extern __shared__ float s[];

    /*локальный индекс потока*/
    int tid = threadIdx.x;
    /*глобальный индекс элемента массива*/
    int gid = blockIdx.x * blockDim.x + tid;

    /*загружаем элемент в shared*/
    float x = (gid < n) ? in[gid] : 0.0f;
    s[tid] = x;
    __syncthreads();

    /*фаза upsweep, суммирование в виде дерева*/
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < blockDim.x) s[idx] += s[idx - offset];
        __syncthreads();
    }

    /*сохраняем сумму блока и делаем exclusive scan*/
    if (tid == 0) {
        if (blockSums) blockSums[blockIdx.x] = s[blockDim.x - 1];
        s[blockDim.x - 1] = 0.0f;
    }
    __syncthreads();

    /*фаза downsweep, формирование exclusive scan*/
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < blockDim.x) {
            float t = s[idx - offset];
            s[idx - offset] = s[idx];
            s[idx] += t;
        }
        __syncthreads();
    }

    /*записываем результат exclusive scan в глобальную память*/
    if (gid < n) out[gid] = s[tid];
}

/*CUDA-ядро, добавление смещений блоков к элементам массива*/
__global__ void add_block_offsets(float* data, const float* blockOffsets, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        int b = gid / blockDim.x;
        data[gid] += blockOffsets[b];
    }
}

/*CUDA-ядро, преобразование exclusive scan в inclusive scan*/
__global__ void exclusive_to_inclusive(const float* in, float* ex, float* inc, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) inc[gid] = ex[gid] + in[gid];
}

/*рекурсивная функция GPU exclusive scan, выполняет сканирование массива произвольного размера*/
void gpu_exclusive_scan(const float* d_in, float* d_out, int n, int threads) {
    int blocks = ceil_div(n, threads);

    /*если массив помещается в один блок*/
    if (blocks == 1) {
        scan_block_exclusive<<<1, threads, threads * sizeof(float)>>>(d_in, d_out, nullptr, n);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    /*выделяем память под суммы блоков и их сканирование*/
    float* d_blockSums = nullptr;
    float* d_blockOffsets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_blockSums, blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_blockOffsets, blocks * sizeof(float)));

    /*выполняем scan для каждого блока*/
    scan_block_exclusive<<<blocks, threads, threads * sizeof(float)>>>(d_in, d_out, d_blockSums, n);
    CUDA_CHECK(cudaGetLastError());

    /*рекурсивно сканируем массив сумм блоков*/
    gpu_exclusive_scan(d_blockSums, d_blockOffsets, blocks, threads);

    /*добавляем смещения блоков ко всем элементам*/
    add_block_offsets<<<blocks, threads>>>(d_out, d_blockOffsets, n);
    CUDA_CHECK(cudaGetLastError());

    /*освобождаем временную память*/
    CUDA_CHECK(cudaFree(d_blockSums));
    CUDA_CHECK(cudaFree(d_blockOffsets));
}

/*последовательная реализация inclusive scan на CPU*/
void cpu_inclusive_scan(const std::vector<float>& in, std::vector<float>& out) {
    out.resize(in.size());
    double acc = 0.0;
    for (size_t i = 0; i < in.size(); i++) {
        acc += in[i];
        out[i] = (float)acc;
    }
}

/*главная функция программы*/
int main() {
    /*размер массива*/
    const int N = 1'000'000;

    /*массив на CPU*/
    std::vector<float> h(N);

    /*генерация входных данных*/
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; i++) h[i] = dist(rng);

    /*вычисление inclusive scan на CPU*/
    std::vector<float> cpu_out;
    auto c0 = std::chrono::high_resolution_clock::now();
    cpu_inclusive_scan(h, cpu_out);
    auto c1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();

    /*выделение памяти на GPU*/
    float *d_in=nullptr, *d_ex=nullptr, *d_inc=nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ex, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_inc, N*sizeof(float)));

    /*копирование данных на GPU*/
    CUDA_CHECK(cudaMemcpy(d_in, h.data(), N*sizeof(float), cudaMemcpyHostToDevice));

    /*параметры запуска CUDA*/
    int threads = 1024;
    int blocks = (N + threads - 1) / threads;

    /*замер времени GPU*/
    cudaEvent_t e0,e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    CUDA_CHECK(cudaEventRecord(e0));

    gpu_exclusive_scan(d_in, d_ex, N, threads);
    exclusive_to_inclusive<<<blocks, threads>>>(d_in, d_ex, d_inc, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));

    float gpu_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, e0, e1));

    /*копирование результата обратно на CPU*/
    std::vector<float> gpu_out(N);
    CUDA_CHECK(cudaMemcpy(gpu_out.data(), d_inc, N*sizeof(float), cudaMemcpyDeviceToHost));

    /*проверка корректности*/
    float max_abs = 0.f;
    for (int i = 0; i < N; i++) {
        float d = std::abs(cpu_out[i] - gpu_out[i]);
        if (d > max_abs) max_abs = d;
    }

    /*вывод результатов*/
    std::cout << "N=" << N << "\n";
    std::cout << "CPU inclusive scan time(ms)=" << cpu_ms << "\n";
    std::cout << "GPU inclusive scan time(ms)=" << gpu_ms << "\n";
    std::cout << "Max abs diff=" << max_abs << "\n";

    /*освобождение памяти GPU*/
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_ex));
    CUDA_CHECK(cudaFree(d_inc));

    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));

    return 0;
}
