/*Задание 2: Распределённое решение системы линейных уравнений методом Гаусса 
Описание задачи: 
Напишите программу для распределённого решения системы линейных уравнений методом Гаусса: 
1. Процесс с "rank = 0" создаёт матрицу коэффициентов A  размером NxN и вектор правых частей b. 
2. Разделите строки матрицы между процессами с помощью функции "MPI_Scatter". 
3. Реализуйте следующие шаги метода Гаусса: 
- Прямой ход: каждый процесс выполняет вычитание строк для своей части матрицы. 
- Обратный ход: соберите результаты на процессе с "rank = 0" и завершите вычисления. 
4. Выведите решение системы уравнений на экран.*/

/*библиотека MPI: инициализация, обмен данными, синхронизация*/
#include <mpi.h>
/*стандартный ввод и вывод*/
#include <iostream>
/*контейнер vector для хранения массивов*/
#include <vector>
/*генерация случайных чисел*/
#include <random>
/*математические функции*/
#include <cmath>
/*std::min и std::max*/
#include <algorithm>

/*функция для удобного доступа к элементам матрицы A*/
static inline double& Aat(std::vector<double>& A, int row, int col, int N) {
    return A[static_cast<size_t>(row) * N + col];
}

/*функция для чтения элементов матрицы A*/
static inline double Aat(const std::vector<double>& A, int row, int col, int N) {
    return A[static_cast<size_t>(row) * N + col];
}

/*главная функция программы*/
int main(int argc, char** argv) {
    /*инициализация MPI*/
    MPI_Init(&argc, &argv);

    /*rank-номер текущего процесса, size-общее количество процессов*/
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /*размер системы линейных уравнений N*/
    int N = 6;
    if (argc >= 2) N = std::stoi(argv[1]);
    if (N <= 0) {
        if (rank == 0) std::cerr << "N must be positive\n";
        MPI_Finalize();
        return 1;
    }

    /*начало замера времени выполнения*/
    double start_time = MPI_Wtime();

    /*чтобы использовать MPI_Scatter, делаем одинаковое количество строк на каждый процесс*/
    int rows_per_proc = (N + size - 1) / size;   /*ceil(N/size)*/
    int padded_rows = rows_per_proc * size;      /*общее число строк с учётом padding*/

    /*матрица коэффициентов A и вектор правых частей b, полностью хранятся только на процессе rank=0*/
    std::vector<double> A;
    std::vector<double> b;

    if (rank == 0) {
        /*выделяем память под матрицу и вектор с учётом padding*/
        A.assign(static_cast<size_t>(padded_rows) * N, 0.0);
        b.assign(static_cast<size_t>(padded_rows), 0.0);

        /*генератор случайных чисел*/
        std::mt19937_64 gen(12345);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        /*создаём диагонально доминируемую матрицу, чтобы метод Гаусса был устойчивым*/
        for (int i = 0; i < N; i++) {
            double rowsum = 0.0;
            for (int j = 0; j < N; j++) {
                if (i == j) continue;
                double v = dist(gen);
                Aat(A, i, j, N) = v;
                rowsum += std::abs(v);
            }
            Aat(A, i, i, N) = rowsum + 1.0;
            b[static_cast<size_t>(i)] = dist(gen);
        }
        /*padding-строки остаются нулевыми и в вычислениях не участвуют*/
    }

    /*локальные части матрицы и вектора для каждого процесса*/
    std::vector<double> localA(static_cast<size_t>(rows_per_proc) * N, 0.0);
    std::vector<double> localb(static_cast<size_t>(rows_per_proc), 0.0);

    /*распределяем строки матрицы A между процессами*/
    MPI_Scatter(rank == 0 ? A.data() : nullptr,
                rows_per_proc * N, MPI_DOUBLE,
                localA.data(),
                rows_per_proc * N, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    /*распределяем элементы вектора b*/
    MPI_Scatter(rank == 0 ? b.data() : nullptr,
                rows_per_proc, MPI_DOUBLE,
                localb.data(),
                rows_per_proc, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    /*буферы для хранения опорной строки*/
    std::vector<double> pivotRow(static_cast<size_t>(N), 0.0);
    double pivotB = 0.0;

    /*прямой ход метода Гаусса*/
    for (int k = 0; k < N; k++) {
        /*определяем процесс, который владеет текущей опорной строкой*/
        int owner = k / rows_per_proc;
        int owner_local_idx = k % rows_per_proc;

        if (rank == owner) {
            /*копируем опорную строку из локального блока*/
            for (int j = 0; j < N; j++)
                pivotRow[static_cast<size_t>(j)] =
                    Aat(localA, owner_local_idx, j, N);
            pivotB = localb[static_cast<size_t>(owner_local_idx)];
        }

        /*передаём опорную строку всем процессам*/
        MPI_Bcast(pivotRow.data(), N, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        MPI_Bcast(&pivotB, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

        double pivot = pivotRow[static_cast<size_t>(k)];
        if (std::abs(pivot) < 1e-14) {
            if (rank == 0) {
                std::cerr << "Zero or near-zero pivot at k=" << k << "\n";
            }
            MPI_Abort(MPI_COMM_WORLD, 2);
        }

        /*обновляем строки ниже опорной*/
        for (int i = 0; i < rows_per_proc; i++) {
            int global_i = rank * rows_per_proc + i;
            if (global_i <= k || global_i >= N) continue;

            double factor = Aat(localA, i, k, N) / pivot;
            for (int j = k; j < N; j++) {
                Aat(localA, i, j, N) -=
                    factor * pivotRow[static_cast<size_t>(j)];
            }
            localb[static_cast<size_t>(i)] -= factor * pivotB;
        }
    }

    /*подготавливаем буферы для сборки результата на rank=0*/
    if (rank == 0) {
        A.assign(static_cast<size_t>(padded_rows) * N, 0.0);
        b.assign(static_cast<size_t>(padded_rows), 0.0);
    }

    /*собираем преобразованную матрицу A*/
    MPI_Gather(localA.data(), rows_per_proc * N, MPI_DOUBLE,
               rank == 0 ? A.data() : nullptr,
               rows_per_proc * N, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    /*собираем преобразованный вектор b*/
    MPI_Gather(localb.data(), rows_per_proc, MPI_DOUBLE,
               rank == 0 ? b.data() : nullptr,
               rows_per_proc, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    /*окончание замера времени*/
    double end_time = MPI_Wtime();

    /*обратный ход выполняется только на процессе rank=0*/
    if (rank == 0) {
        std::vector<double> x(static_cast<size_t>(N), 0.0);

        for (int i = N - 1; i >= 0; i--) {
            double sum = 0.0;
            for (int j = i + 1; j < N; j++) {
                sum += Aat(A, i, j, N) * x[static_cast<size_t>(j)];
            }
            x[static_cast<size_t>(i)] =
                (b[static_cast<size_t>(i)] - sum) / Aat(A, i, i, N);
        }

        /*вывод решения системы*/
        std::cout << "N = " << N << "\n";
        std::cout << "Processes = " << size << "\n";
        std::cout << "rows_per_proc = " << rows_per_proc
                  << " (padded_rows = " << padded_rows << ")\n";
        std::cout << "Solution x:\n";
        for (int i = 0; i < N; i++) {
            std::cout << "x[" << i << "] = "
                      << x[static_cast<size_t>(i)] << "\n";
        }
        std::cout << "Execution time: "
                  << (end_time - start_time) << " seconds\n";
    }

    /*завершение работы MPI*/
    MPI_Finalize();
    return 0;
}
