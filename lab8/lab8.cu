/*Практическая работа №8
Задание 1: Реализация обработки массива на CPU с использованием OpenMP 
1. Создайте массив данных размером `N` (например, `N = 1 000 000`). 
2. Реализуйте функцию для обработки массива на CPU с использованием 
OpenMP. Например, умножьте каждый элемент массива на 2. 
3. Замерьте время выполнения обработки на CPU. 
Задание 2: Реализация обработки массива на GPU с использованием CUDA 
1. Скопируйте массив данных на GPU. 
2. Реализуйте ядро CUDA для обработки массива на GPU. Например, умножьте 
каждый элемент массива на 2. 
3. Скопируйте обработанные данные обратно на CPU. 
4. Замерьте время выполнения обработки на GPU. 
Задание 3: Гибридная обработка массива 
1. Разделите массив на две части: первая половина обрабатывается на CPU, 
вторая — на GPU. 
2. Реализуйте гибридное приложение, которое выполняет обработку массива 
на CPU и GPU одновременно. 
3. Замерьте общее время выполнения гибридной обработки. 
Задание 4: Анализ производительности 
1. Сравните время выполнения обработки массива на CPU, GPU и в гибридном 
режиме. 
2. Проведите анализ производительности и определите, в каких случаях 
гибридный подход дает наибольший выигрыш. */

/*подключаем стандартные библиотеки*/
#include <cstdio>      /*printf*/
#include <cstdlib>     /*exit, atoi*/
#include <vector>      /*std::vector*/
#include <chrono>      /*измерение времени*/
#include <iostream>    /*std::cout*/
#include <cassert>     /*assert*/

/*подключаем CUDA Runtime API*/
#include <cuda_runtime.h>

/*подключаем OpenMP при наличии*/
#ifdef _OPENMP
#include <omp.h>
#endif

/*макрос для проверки ошибок CUDA*/
#define CUDA_CHECK(call) do {                                 \
  cudaError_t err = (call);                                   \
  if (err != cudaSuccess) {                                   \
    std::cerr << "CUDA error: " << cudaGetErrorString(err)     \
              << " at " << __FILE__ << ":" << __LINE__ << "\n";\
    std::exit(1);                                             \
  }                                                           \
} while(0)

/*CUDA ядро: каждый поток умножает один элемент массива на 2*/
__global__ void mul2_kernel(float* data, int n) {
    /*глобальный индекс элемента*/
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    /*проверка выхода за границы массива*/
    if (i < n) data[i] *= 2.0f;
}

/*функция обработки массива на CPU с использованием OpenMP*/
void mul2_openmp(float* data, int n) {
    /*распараллеливаем цикл for*/
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
        data[i] *= 2.0f;
}

/*функция для измерения времени выполнения на CPU*/
double seconds_since(const std::chrono::high_resolution_clock::time_point& t0) {
    using namespace std::chrono;
    return duration_cast<duration<double>>(high_resolution_clock::now() - t0).count();
}

/*функция для проверки корректности результатов, возвращает максимальное абсолютное отклонение*/
float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    assert(a.size() == b.size());
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float d = std::abs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

/*главная функция программы*/
int main(int argc, char** argv) {

    /*размер массива*/
    int N = 1'000'000;
    if (argc >= 2) N = std::atoi(argv[1]);

    std::cout << "N = " << N << "\n";

#ifdef _OPENMP
    std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";
#else
    std::cout << "OpenMP not enabled\n";
#endif

    /*инициализация исходного массива на CPU*/
    std::vector<float> base(N);
    for (int i = 0; i < N; i++)
        base[i] = float(i) * 0.001f + 1.0f;

    /*формирование эталонного результата*/
    std::vector<float> ref = base;
    for (int i = 0; i < N; i++)
        ref[i] *= 2.0f;

    /*Задание 1: обработка массива на CPU с OpenMP*/
    std::vector<float> cpu = base;
    auto t0 = std::chrono::high_resolution_clock::now();
    mul2_openmp(cpu.data(), N);
    double t_cpu = seconds_since(t0);

    std::cout << "\nCPU(OpenMP) time: "
              << (t_cpu * 1000.0) << " ms\n";
    std::cout << "Diff vs ref: "
              << max_abs_diff(cpu, ref) << "\n";

    /*Задание 2: обработка массива на GPU с использованием CUDA*/
    std::vector<float> gpu = base;
    float* d = nullptr;

    /*выделяем память на GPU*/
    CUDA_CHECK(cudaMalloc(&d, N * sizeof(float)));

    /*события CUDA для измерения времени*/
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    /*начало замера времени*/
    CUDA_CHECK(cudaEventRecord(e0));

    /*копирование данных с CPU на GPU*/
    CUDA_CHECK(cudaMemcpy(d, gpu.data(),
                           N * sizeof(float),
                           cudaMemcpyHostToDevice));

    /*настройка конфигурации запуска ядра*/
    int block = 256;
    int grid = (N + block - 1) / block;

    /*запуск CUDA ядра*/
    mul2_kernel<<<grid, block>>>(d, N);
    CUDA_CHECK(cudaGetLastError());

    /*копирование результата обратно на CPU*/
    CUDA_CHECK(cudaMemcpy(gpu.data(),
                           d,
                           N * sizeof(float),
                           cudaMemcpyDeviceToHost));

    /*окончание замера времени*/
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));

    float ms_gpu = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_gpu, e0, e1));

    std::cout << "\nGPU(CUDA) time (H2D + kernel + D2H): "
              << ms_gpu << " ms\n";
    std::cout << "Diff vs ref: "
              << max_abs_diff(gpu, ref) << "\n";

    /*освобождаем ресурсы GPU*/
    CUDA_CHECK(cudaFree(d));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));

    /*Задание 3: гибридная обработка CPU+GPU*/
    std::vector<float> hybrid = base;

    /*разделяем массив на две части*/
    int n_cpu = N / 2;
    int n_gpu = N - n_cpu;

    float* d2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d2, n_gpu * sizeof(float)));

    /*события для измерения GPU части гибридного варианта*/
    cudaEvent_t g0, g1;
    CUDA_CHECK(cudaEventCreate(&g0));
    CUDA_CHECK(cudaEventCreate(&g1));

    auto th0 = std::chrono::high_resolution_clock::now();

    /*асинхронно запускаем обработку второй половины на GPU*/
    CUDA_CHECK(cudaEventRecord(g0));
    CUDA_CHECK(cudaMemcpyAsync(d2,
                               hybrid.data() + n_cpu,
                               n_gpu * sizeof(float),
                               cudaMemcpyHostToDevice));

    int grid2 = (n_gpu + block - 1) / block;
    mul2_kernel<<<grid2, block>>>(d2, n_gpu);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(hybrid.data() + n_cpu,
                               d2,
                               n_gpu * sizeof(float),
                               cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(g1));

    /*одновременно обрабатываем первую половину массива на CPU*/
    mul2_openmp(hybrid.data(), n_cpu);

    /*ожидаем завершения GPU вычислений*/
    CUDA_CHECK(cudaEventSynchronize(g1));

    double t_hybrid = seconds_since(th0);

    float ms_gpu_part = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_gpu_part, g0, g1));

    std::cout << "\nHybrid total wall-time: "
              << (t_hybrid * 1000.0) << " ms\n";
    std::cout << "         GPU part time (async, incl copies): "
              << ms_gpu_part << " ms\n";
    std::cout << "Diff vs ref: "
              << max_abs_diff(hybrid, ref) << "\n";

    /*освобождаем ресурсы*/
    CUDA_CHECK(cudaFree(d2));
    CUDA_CHECK(cudaEventDestroy(g0));
    CUDA_CHECK(cudaEventDestroy(g1));

    /*Задание 4: анализ полученных результатов*/
    /*выводим сводную информацию по времени выполнения*/
    std::cout << "\nSummary:\n";
    std::cout << "CPU(OpenMP): " << (t_cpu * 1000.0) << " ms\n";
    std::cout << "GPU(CUDA):   " << ms_gpu << " ms (H2D + kernel + D2H)\n";
    std::cout << "Hybrid:      " << (t_hybrid * 1000.0) << " ms (wall-time)\n";

    /*завершение программы*/
    return 0;
}
