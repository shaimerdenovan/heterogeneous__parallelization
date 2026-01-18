/*Assignment 3. Архитектура GPU и оптимизация CUDA-программ
Задание 2
Реализуйте CUDA-программу для поэлементного сложения двух массивов. Исследуйте
влияние размера блока потоков на производительность программы. Проведите замеры
времени для как минимум трёх различных размеров блока.*/

/*подключаем библиотеку CUDA Runtime API*/
#include <cuda_runtime.h>
/*библиотека для вывода printf*/
#include <cstdio>
/*библиотека для exit*/
#include <cstdlib>
/*библиотека vector для хранения данных на CPU*/
#include <vector>
/*могут пригодиться для расширения/проверок*/
#include <cmath>
#include <algorithm>

/*макрос для проверки ошибок CUDA*/
#define CUDA_CHECK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} }while(0)

/*CUDA-ядро:поэлементное сложение двух массивов*/
/*каждый поток вычисляет один элемент: c[i] = a[i] + b[i]*/
__global__ void add_vec(float* c, const float* a, const float* b, int n){
  /*глобальный индекс элемента*/
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  /*проверка выхода за границы массива*/
  if(i < n) c[i] = a[i] + b[i];
}

/*функция замера времени для заданного размера блока*/
/*N-размер массивов*/
/*block-количество потоков в блоке*/
/*warmup-прогрев, iters-число повторов для усреднения*/
float time_add(int N, int block, int warmup=5, int iters=100){
  /*объём памяти под один массив*/
  size_t bytes = (size_t)N*sizeof(float);

  /*создаём входные массивы на CPU*/
  std::vector<float> h_a(N), h_b(N);

  /*заполняем входные массивы значениями*/
  for(int i=0;i<N;i++){
    h_a[i] = 2.0f + 0.001f*(i%1000);
    h_b[i] = 3.0f + 0.001f*(i%1000);
  }

  /*указатели на массивы в памяти GPU*/
  float *d_a, *d_b, *d_c;

  /*выделяем память на GPU под a, b и c*/
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));

  /*копируем входные данные с CPU на GPU*/
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

  /*задаём конфигурацию запуска ядра: B-размер блока, G-количество блоков*/
  dim3 B(block);
  dim3 G((N + block - 1)/block);

  /*прогрев:несколько запусков до замера времени*/
  for(int i=0;i<warmup;i++) add_vec<<<G,B>>>(d_c,d_a,d_b,N);
  CUDA_CHECK(cudaDeviceSynchronize());

  /*замер времени через cudaEvent*/
  cudaEvent_t st,en;
  CUDA_CHECK(cudaEventCreate(&st));
  CUDA_CHECK(cudaEventCreate(&en));

  /*старт*/
  CUDA_CHECK(cudaEventRecord(st));
  /*повторы*/
  for(int i=0;i<iters;i++) add_vec<<<G,B>>>(d_c,d_a,d_b,N);
  /*финиш*/
  CUDA_CHECK(cudaEventRecord(en));
  CUDA_CHECK(cudaEventSynchronize(en));

  /*время в миллисекундах*/
  float ms;
  CUDA_CHECK(cudaEventElapsedTime(&ms, st, en));

  /*удаляем события*/
  CUDA_CHECK(cudaEventDestroy(st));
  CUDA_CHECK(cudaEventDestroy(en));

  /*освобождаем память GPU*/
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  /*возвращаем среднее время на один запуск ядра*/
  return ms/iters;
}

/*главная функция программы*/
int main(){
  /*размер массивов по условию задания*/
  const int N = 1'000'000;

  /*набор размеров блока*/
  int blocks[] = {128, 256, 512};

  /*выводим заголовок*/
  printf("Task2 N=%d (vector add)\n", N);

  /*для каждого размера блока измеряем время работы ядра*/
  for(int b: blocks){
    float t = time_add(N, b);
    printf("block=%4d -> %.4f ms\n", b, t);
  }

  /*завершение программы*/
  return 0;
}
