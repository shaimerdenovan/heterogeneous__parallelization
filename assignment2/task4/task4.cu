/*Задача 4. Сортировка на GPU с использованием CUDA
Практическое задание
Реализуйте параллельную сортировку слиянием на GPU с использованием CUDA:
-разделите массив на подмассивы, каждый из которых обрабатывается
отдельным блоком;
-выполните параллельное слияние отсортированных подмассивов;
-замерьте производительность для массивов размером 10 000 и 100 000
элементов.*/
/*подключаем заголовочный файл для работы с CUDA runtime API*/
#include <cuda_runtime.h>
/*подключаем стандартную библиотеку ввода и вывода*/
#include <iostream>
/*подключаем библиотеку алгоритмов для std::min и std::is_sorted*/
#include <algorithm>
/*подключаем библиотеку для работы с векторами*/
#include <vector>
/*подключаем библиотеку для измерения времени*/
#include <chrono>

/*определяем количество потоков в блоке*/
#define THREADS_PER_BLOCK 512

/*CUDA kernel для слияния*/
/*функция ядра mergeKernel выполняет слияние двух отсортированных подмассивов*/
__global__ void mergeKernel(int* arr, int* tmp, int width, int n) {
    /*глобальный индекс потока в сетке CUDA*/
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    /*начальный индекс подмассива для данного потока*/
    int start = idx * (2 * width);
    /*если начало подмассива выходит за пределы массива, выходим*/
    if(start >= n) return;

    /*средняя граница подмассива*/
    int mid = std::min(start + width, n);
    /*конец подмассива (не превышаем размер массива)*/
    int end = std::min(start + 2 * width, n);

    /*индексы для прохода по двум подмассивам и временный индекс*/
    int i = start, j = mid, k = start;
    /*слияние элементов двух подмассивов в tmp*/
    while(i < mid && j < end) {
        tmp[k++] = (arr[i] <= arr[j]) ? arr[i++] : arr[j++];
    }
    /*если остались элементы в первом подмассиве, копируем их*/
    while(i < mid) tmp[k++] = arr[i++];
    /*если остались элементы во втором подмассиве, копируем их*/
    while(j < end) tmp[k++] = arr[j++];
}

/*функция CUDA merge sort*/
/*выполняет многопроходную сортировку слиянием на GPU*/
void mergeSortGPU(int* arr, int n) {
    /*выделяем память на устройстве GPU для основного массива и временного массива*/
    int* d_arr;
    int* d_tmp;
    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMalloc(&d_tmp, n * sizeof(int));
    /*копируем данные с хоста на устройство GPU*/
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    /*width - текущий размер подмассива, удваивается на каждом проходе*/
    for(int width = 1; width < n; width *= 2) {
        /*вычисляем количество блоков, необходимых для обработки всех подмассивов*/
        int blocks = (n + (2*width*THREADS_PER_BLOCK - 1)) / (2*width*THREADS_PER_BLOCK);
        /*запускаем ядро mergeKernel с вычисленным числом блоков и потоков*/
        mergeKernel<<<blocks, THREADS_PER_BLOCK>>>(d_arr, d_tmp, width, n);
        /*синхронизация устройства, чтобы убедиться, что все блоки завершили работу*/
        cudaDeviceSynchronize();
        /*меняем местами массивы для следующего шага слияния*/
        std::swap(d_arr, d_tmp);
    }

    /*копируем отсортированный массив обратно на хост*/
    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    /*освобождаем память на GPU*/
    cudaFree(d_arr);
    cudaFree(d_tmp);
}

/*главная функция*/
int main() {
    /*массив размеров для тестирования (10000 и 100000 элементов)*/
    std::vector<int> sizes = {10000, 100000};

    /*цикл по всем размерам массивов*/
    for(int n : sizes) {
        /*выделяем память на хосте для массива*/
        std::vector<int> arr(n);
        /*заполняем массив случайными числами от 0 до 99999*/
        for(int i = 0; i < n; i++) arr[i] = rand() % 100000;

        /*измеряем время выполнения сортировки на GPU*/
        auto start = std::chrono::high_resolution_clock::now();
        mergeSortGPU(arr.data(), n);
        auto end = std::chrono::high_resolution_clock::now();

        /*вычисляем продолжительность в миллисекундах*/
        double duration = std::chrono::duration<double, std::milli>(end - start).count();
        /*выводим результат*/
        std::cout << "Sorted " << n << " elements in " << duration << " ms" << std::endl;

        /*проверка, что массив отсортирован*/
        if(!std::is_sorted(arr.begin(), arr.end())) {
            std::cout << "Error: Array is not sorted!" << std::endl;
        }
    }

    /*завершение программы*/
    return 0;
}