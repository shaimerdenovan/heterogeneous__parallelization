/*Практическая работа 6: Программирование на OpenCL для CPU и GPU
Задача №1.
1. Подготовка окружения:
1. Установите необходимые драйверы для OpenCL на вашей системе.
a. Для CPU: установите драйверы от производителя процессора
(Intel, AMD).
b. Для GPU: установите драйверы от производителя видеокарты
(NVIDIA, AMD).
2. Убедитесь, что OpenCL-библиотеки доступны в вашей системе:
a. Linux: /usr/lib/libOpenCL.so
b. Windows: OpenCL.dll
3. Настройте среду разработки (например, VS Code, CLion или Visual
Studio).
4. 2. Реализация задачи
Задача №2.
Описание задачи:
Реализуйте программу для параллельного умножения двух матриц с
использованием OpenCL. Матрицы A и B имеют размеры N×M и M×K
соответственно. Программа должна вычислить результирующую матрицу C
размером N×K.*/


/*Задача №1: Vector Add
C[i] = A[i] + B[i]
Один work-item -один элемент массива*/
__kernel void vector_add(__global const float* A, /*входной массив A в global memory*/
                         __global const float* B, /*входной массив B в global memory*/
                         __global float* C,       /*выходной массив C в global memory*/
                         const int n)             /*размер векторов (число элементов)*/
{
    /*получаем глобальный индекс work-item*/
    int id = get_global_id(0);

    /*проверка на выход за границы массива
      нужна, если global size округляют вверх или задают больше n*/
    if (id < n) {
        /*элементное сложение*/
        C[id] = A[id] + B[id];
    }
}


/*Задача №2: Matrix Multiply
C = A × B
A: размер N×M
B: размер M×K
C: размер N×K
Каждый work-item вычисляет один элемент C[row, col]:
C[row, col] = Σ(A[row, i] * B[i, col]), i=0..M-1*/
__kernel void mat_mul(__global const float* A, /*матрица A (N×M) в global memory*/
                      __global const float* B, /*матрица B (M×K) в global memory*/
                      __global float* C,       /*матрица C (N×K) в global memory*/
                      const int N,             /*число строк A и C*/
                      const int M,             /*число столбцов A и строк B*/
                      const int K)             /*число столбцов B и C*/
{
    /*координаты элемента результирующей матрицы*/
    int col = get_global_id(0); /*индекс столбца (0..K-1)*/
    int row = get_global_id(1); /*индекс строки   (0..N-1)*/

    /*проверка выхода за границы*/
    if (row < N && col < K) {

        /*аккумулятор суммы*/
        float sum = 0.0f;

        /*суммируем произведения по общей размерности M*/
        for (int i = 0; i < M; i++) {
            /*A[row, i] хранится как A[row*M + i]
              B[i, col] хранится как B[i*K + col]*/
            sum += A[row * M + i] * B[i * K + col];
        }

        /*записываем результат в C[row, col] - C[row*K + col]*/
        C[row * K + col] = sum;
    }
}
