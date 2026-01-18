/*Assignment 3. Архитектура GPU и оптимизация CUDA-программ
Задание 1
Реализуйте программу на CUDA для поэлементной обработки массива (например,
умножение каждого элемента на число). Реализуйте две версии программы:
1. с использованием только глобальной памяти;
2. с использованием разделяемой памяти.
Сравните время выполнения обеих реализаций для массива размером 1 000 000
элементов.*/

/*подключаем библиотеку CUDA Runtime API*/
#include <cuda_runtime.h>
/*библиотека для вывода printf*/
#include <cstdio>
/*библиотека для exit*/
#include <cstdlib>
/*библиотека vector для хранения данных на CPU*/
#include <vector>
/*библиотека cmath для fabs*/
#include <cmath>
/*библиотека algorithm для std::max*/
#include <algorithm>

/*макрос для проверки ошибок CUDA (если ошибка то печать и завершение программы)*/
#define CUDA_CHECK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} }while(0)

/*1: умножение через глобальную память*/
/*каждый поток обрабатывает один элемент массива*/
__global__ void mul_global(float* out, const float* in, float k, int n){
  /*глобальный индекс элемента=индекс блока*размер блока+индекс потока*/
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  /*проверяем границы массива*/
  if(i < n) out[i] = in[i] * k;
}

/*2: умножение с использованием разделяемой памяти*/
/*сначала копируем элемент из глобальной в разделяемую, затем умножаем и пишем в глобальную*/
/*для этой простой операции разделяемая обычно не ускоряет, потому что добавляет syncthreads*/
__global__ void mul_shared(float* out, const float* in, float k, int n){
  /*динамическая разделяемая память,выделяется при запуске ядра*/
  extern __shared__ float s[];

  /*глобальный индекс элемента*/
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  /*локальный индекс в блоке*/
  int tid = threadIdx.x;

  /*если индекс в пределах массива то загружаем из глобальной в разделяемую*/
  if(gid < n) s[tid] = in[gid];

  /*барьер синхронизации,все потоки должны завершить запись в разделяемую*/
  __syncthreads();

  /*после синхронизации выполняем умножение и запись результата в глобальную*/
  if(gid < n) out[gid] = s[tid] * k;
}

/*функция замера времени выполнения ядра через cudaEvent*/
/*прогрев чтобы исключить влияние первого запуска и JIT*/
/*iters-число повторов для усреднения*/
/*shmem-сколько байт shared памяти выделять на блок*/
/*kernel-указатель на ядро*/
float time_kernel(dim3 G, dim3 B, size_t shmem, int warmup, int iters,
                  void(*kernel)(float*, const float*, float, int),
                  float* out, const float* in, float k, int n){
  /*прогрев ядра*/
  for(int i=0;i<warmup;i++) kernel<<<G,B,shmem>>>(out,in,k,n);
  /*ждём завершения прогрева*/
  CUDA_CHECK(cudaDeviceSynchronize());

  /*создаём события для замера времени на GPU*/
  cudaEvent_t st, en;
  CUDA_CHECK(cudaEventCreate(&st));
  CUDA_CHECK(cudaEventCreate(&en));

  /*записываем стартовое событие*/
  CUDA_CHECK(cudaEventRecord(st));

  /*многократно запускаем ядро для усреднения*/
  for(int i=0;i<iters;i++) kernel<<<G,B,shmem>>>(out,in,k,n);

  /*записываем конечное событие*/
  CUDA_CHECK(cudaEventRecord(en));
  /*ждём пока конечное событие будет достигнуто,то есть все ядра завершатся*/
  CUDA_CHECK(cudaEventSynchronize(en));

  /*получаем время в миллисекундах*/
  float ms;
  CUDA_CHECK(cudaEventElapsedTime(&ms, st, en));

  /*удаляем события*/
  CUDA_CHECK(cudaEventDestroy(st));
  CUDA_CHECK(cudaEventDestroy(en));

  /*возвращаем среднее время на один запуск*/
  return ms / iters;
}

/*главная функция программы*/
int main(){
  /*размер массива по условию задания*/
  const int N = 1'000'000;
  /*объём памяти в байтах*/
  const size_t bytes = (size_t)N * sizeof(float);

  /*создаём массивы на CPU: входной, выходной и эталонный*/
  std::vector<float> h_in(N), h_out(N), ref(N);

  /*заполняем входной массив данными*/
  for(int i=0;i<N;i++) h_in[i] = 1.0f + 0.001f*(i%1000);

  /*указатели на память GPU*/
  float *d_in=nullptr, *d_out=nullptr;

  /*выделяем память на GPU под вход и выход*/
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));

  /*копируем входной массив с CPU на GPU*/
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

  /*задаём размер блока потоков*/
  int block = 256;

  /*формируем конфигурацию запуска:
    B-количество потоков в блоке,
    G-количество блоков в сетке*/
  dim3 B(block);
  dim3 G((N + block - 1)/block);

  /*множитель для умножения элементов*/
  float k = 1.2345f;

  /*параметры замеров*/
  int warmup=5;
  int iters=50;

  /*замер версии 1 только глобальня память*/
  float t_glob = time_kernel(G,B,0,warmup,iters,
      (void(*)(float*,const float*,float,int))mul_global,
      d_out,d_in,k,N);

  /*замер версии 2 с разделяемой памятью*/
  /*выделяем shared память: block * sizeof(float) на блок*/
  float t_sh   = time_kernel(G,B,block*sizeof(float),warmup,iters,
      (void(*)(float*,const float*,float,int))mul_shared,
      d_out,d_in,k,N);

  /*проверка корректности*/
  /*копируем результат обратно на CPU*/
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

  /*формируем эталонный результат на CPU*/
  for(int i=0;i<N;i++) ref[i] = h_in[i]*k;

  /*находим максимальную абсолютную ошибку*/
  float diff=0.0f;
  for(int i=0;i<N;i++) diff = std::max(diff, std::fabs(h_out[i]-ref[i]));

  /*вывод результатов*/
  printf("Task1 N=%d block=%d\n", N, block);
  printf("Global: %.4f ms\n", t_glob);
  printf("Shared: %.4f ms (shmem/block=%zu bytes)\n", t_sh, block*sizeof(float));
  printf("Max abs diff: %g\n", diff);

  /*освобождаем память на GPU*/
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));

  /*завершение программы*/
  return 0;
}
