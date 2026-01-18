/*Assignment 3. Архитектура GPU и оптимизация CUDA-программ
Задание 3
Реализуйте CUDA-программу для обработки массива, демонстрирующую
коалесцированный и некоалесцированный доступ к глобальной памяти. Сравните время
выполнения обеих реализаций для массива размером 1 000 000 элементов.*/

/*подключаем библиотеку CUDA Runtime API*/
#include <cuda_runtime.h>
/*библиотека для вывода printf*/
#include <cstdio>
/*библиотека для exit*/
#include <cstdlib>
/*библиотека vector для хранения данных на CPU*/
#include <vector>

/*макрос для проверки ошибок CUDA*/
#define CUDA_CHECK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} }while(0)

/*версия 1: коалесцированный доступ*/
/*каждый поток читает in[gid], где gid идут подряд в пределах варпа*/
/*то есть потоки варпа читают соседние адреса-память обслуживается эффективно*/
__global__ void coalesced(float* out, const float* in, int n){
  /*глобальный индекс*/
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  /*проверка границ*/
  if(gid < n) out[gid] = in[gid] + 1.0f;
}

/*версия 2:некалесцированный доступ*/
/*каждый поток читает in[idx], где idx вычисляется с шагом stride*/
/*при stride=32 потоки одного варпа обращаются к разнесённым адресам*/
/*это приводит к большему числу транзакций памяти и снижению производительности*/
__global__ void noncoalesced(float* out, const float* in, int n, int stride){
  /*глобальный индекс потока*/
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  /*проверка границ*/
  if(gid < n){
    /*плохой индекс чтения: разнесённые адреса,ломаем коалесцирование*/
    int idx = (int)((1LL * gid * stride) % n);
    /*пишем результат в out[gid], чтобы запись была одинаковой, а различие было именно в чтении*/
    out[gid] = in[idx] + 1.0f;
  }
}

/*функция замера времени выполнения:
noncoal=false-запускаем coalesced, noncoal=true-запускаем noncoalesced, stride используется только для noncoalesced
warmup-прогрев, iters-число повторов для усреднения*/
float time_copy(bool noncoal, int N, int block, int stride=32, int warmup=5, int iters=100){
  /*объём памяти под один массив*/
  size_t bytes = (size_t)N*sizeof(float);

  /*создаём входной массив на CPU*/
  std::vector<float> h_in(N, 1.0f);

  /*указатели на массивы в памяти GPU*/
  float *d_in, *d_out;

  /*выделяем память на GPU*/
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));

  /*копируем входные данные с CPU на GPU*/
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

  /*настраиваем конфигурацию запуска*/
  dim3 B(block);
  dim3 G((N + block - 1)/block);

  /*прогрев, несколько запусков до основного замера*/
  for(int i=0;i<warmup;i++){
    if(noncoal) noncoalesced<<<G,B>>>(d_out,d_in,N,stride);
    else        coalesced<<<G,B>>>(d_out,d_in,N);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  /*замер времени через cudaEvent*/
  cudaEvent_t st,en;
  CUDA_CHECK(cudaEventCreate(&st));
  CUDA_CHECK(cudaEventCreate(&en));

  /*старт*/
  CUDA_CHECK(cudaEventRecord(st));

  /*повторы*/
  for(int i=0;i<iters;i++){
    if(noncoal) noncoalesced<<<G,B>>>(d_out,d_in,N,stride);
    else        coalesced<<<G,B>>>(d_out,d_in,N);
  }

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
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));

  /*возвращаем среднее время на один запуск ядра*/
  return ms/iters;
}

/*главная функция программы*/
int main(){
  /*размер массива по условию задания*/
  const int N = 1'000'000;

  /*выбираем размер блока*/
  int block = 256;

  /*stride для некоалесцированного чтения: 
  stride=32 обычно хорошо демонстрирует ухудшение так как соответствует размеру варпа*/
  int stride = 32;

  /*измеряем время коалесцированной версии*/
  float t1 = time_copy(false, N, block);

  /*измеряем время некоалесцированной версии*/
  float t2 = time_copy(true,  N, block, stride);

  /*вывод результатов*/
  printf("Task3 N=%d block=%d\n", N, block);
  printf("Coalesced:     %.4f ms\n", t1);
  printf("Non-coalesced: %.4f ms (stride=%d)\n", t2, stride);

  /*завершение программы*/
  return 0;
}
