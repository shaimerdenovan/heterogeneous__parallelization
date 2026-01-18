/*Assignment 3. Архитектура GPU и оптимизация CUDA-программ
Задание 4
Для одной из реализованных в предыдущих заданиях CUDA-программ подберите
оптимальные параметры конфигурации сетки и блоков потоков. Сравните
производительность неоптимальной и оптимизированной конфигураций.*/

/*подключаем библиотеку CUDA Runtime API*/
#include <cuda_runtime.h>
/*библиотека для вывода printf*/
#include <cstdio>
/*библиотека для exit*/
#include <cstdlib>
/*библиотека vector для хранения данных на CPU*/
#include <vector>
/*библиотека algorithm для std::remove_if*/
#include <algorithm>

/*макрос для проверки ошибок CUDA*/
#define CUDA_CHECK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} }while(0)

/*CUDA-ядро: поэлементное сложение двух массивов*/
/*каждый поток обрабатывает один элемент массива*/
__global__ void add_vec(float* c, const float* a, const float* b, int n){
  /*глобальный индекс элемента*/
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  /*проверка выхода за границы массива*/
  if(i < n) c[i] = a[i] + b[i];
}

/*функция замера времени выполнения ядра для заданного размера блока*/
/*N-размер массивов*/
/*block-число потоков в блоке*/
/*warmup-прогрев, iters-число повторов для усреднения*/
float time_add(int N, int block, int warmup=5, int iters=100){
  /*объём памяти под один массив*/
  size_t bytes = (size_t)N*sizeof(float);

  /*создаём входные массивы на CPU*/
  std::vector<float> h_a(N, 2.0f);
  std::vector<float> h_b(N, 3.0f);

  /*указатели на массивы в памяти GPU*/
  float *d_a, *d_b, *d_c;

  /*выделяем память на GPU под a, b и c*/
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));

  /*копируем входные данные с CPU на GPU*/
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

  /*настраиваем конфигурацию запуска ядра*/
  dim3 B(block);
  dim3 G((N + block - 1)/block);

  /*прогрев ядра*/
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

  /*получаем время в миллисекундах*/
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

  /*неоптимальная конфигурация, маленький размер блока приводит к меньшей загрузке GPU*/
  int bad_block = 64;
  float t_bad = time_add(N, bad_block);

  /*набор кандидатов для подбора оптимального размера блока*/
  /*учитываем аппаратное ограничение maxThreadsPerBlock*/
  std::vector<int> cand = {64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 768, 1024};

  /*получаем свойства устройства*/
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));

  /*удаляем варианты, превышающие максимальный размер блока*/
  cand.erase(
    std::remove_if(cand.begin(), cand.end(),
                   [&](int x){ return x > p.maxThreadsPerBlock; }),
    cand.end()
  );

  /*поиск оптимального размера блока*/
  float best_t = 1e9f;
  int best_block = -1;

  for(int b: cand){
    float t = time_add(N, b);
    /*сохраняем минимальное время и соответствующий размер блока*/
    if(t < best_t){
      best_t = t;
      best_block = b;
    }
  }

  /*вывод результатов*/
  printf("Task4 N=%d\n", N);
  printf("Bad config:  block=%d -> %.4f ms\n", bad_block, t_bad);
  printf("Best config: block=%d -> %.4f ms\n", best_block, best_t);
  printf("Speedup: %.2fx\n", t_bad / best_t);

  /*завершение программы*/
  return 0;
}
