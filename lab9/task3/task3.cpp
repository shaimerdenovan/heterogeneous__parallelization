/*Задание 3: Параллельный анализ графов (поиск кратчайших путей) 
Описание задачи: 
Напишите программу для параллельного поиска кратчайших путей в графе с использованием алгоритма Флойда-Уоршелла: 
1. Процесс с "rank = 0" создаёт матрицу смежности графа G размером NxN. 
2. Разделите строки матрицы между процессами с помощью функции "MPI_Scatter". 
3. Реализуйте алгоритм Флойда-Уоршелла: 
- Каждый процесс обновляет свою часть матрицы для текущей итерации. 
- Передайте обновлённые данные между процессами с помощью функции "MPI_Allgather". 
4. После завершения всех итераций соберите матрицу на процессе с "rank = 0" и выведите её на экран.*/

/*библиотека MPI: обмен данными, синхронизация, замер времени*/
#include <mpi.h>
/*стандартный ввод и вывод*/
#include <iostream>
/*контейнер vector для хранения матриц*/
#include <vector>
/*генерация случайных чисел*/
#include <random>
/*вспомогательные алгоритмы*/
#include <algorithm>
/*числовые пределы*/
#include <limits>

/*функция доступа к элементам матрицы*/
static inline double& Mat(std::vector<double>& M, int r, int c, int N) {
    return M[static_cast<size_t>(r) * N + c];
}

/*функция доступа к элементам матрицы*/
static inline double Mat(const std::vector<double>& M, int r, int c, int N) {
    return M[static_cast<size_t>(r) * N + c];
}

/*главная функция программы*/
int main(int argc, char** argv) {
    /*инициализация MPI*/
    MPI_Init(&argc, &argv);

    /*rank-номер текущего процесса, size-общее количество процессов*/
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /*размер графа N (число вершин)*/
    int N = 6;
    if (argc >= 2) N = std::stoi(argv[1]);
    if (N <= 0) {
        if (rank == 0) std::cerr << "N must be positive\n";
        MPI_Finalize();
        return 1;
    }

    /*константа для обозначения отсутствия ребра*/
    const double INF = 1e15;

    /*начало замера времени*/
    double start_time = MPI_Wtime();

    /*чтобы использовать MPI_Scatter, делаем одинаковое количество строк на процесс*/
    int rows_per_proc = (N + size - 1) / size;   /*ceil(N/size)*/
    int padded_rows = rows_per_proc * size;      /*общее число строк с padding*/

    /*матрица смежности графа, полностью хранится только на процессе rank=0*/
    std::vector<double> G;
    if (rank == 0) {
        /*выделяем память под матрицу с учётом padding*/
        G.assign(static_cast<size_t>(padded_rows) * N, INF);

        /*генераторы случайных чисел*/
        std::mt19937_64 gen(12345);
        std::uniform_int_distribution<int> wdist(1, 9);
        std::bernoulli_distribution edge(0.7); /*вероятность наличия ребра*/

        /*заполняем реальные строки матрицы*/
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (i == j) {
                    Mat(G, i, j, N) = 0.0;
                } else {
                    Mat(G, i, j, N) =
                        edge(gen) ? static_cast<double>(wdist(gen)) : INF;
                }
            }
        }

        /*padding-строки остаются фиктивными и в вычислениях не участвуют*/
        for (int i = N; i < padded_rows; i++) {
            for (int j = 0; j < N; j++) {
                Mat(G, i, j, N) = INF;
            }
        }
    }

    /*локальная часть матрицы для каждого процесса*/
    std::vector<double> localG(static_cast<size_t>(rows_per_proc) * N, INF);

    /*распределяем строки матрицы между процессами*/
    MPI_Scatter(rank == 0 ? G.data() : nullptr,
                rows_per_proc * N, MPI_DOUBLE,
                localG.data(),
                rows_per_proc * N, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    /*буфер для хранения полной матрицы на каждом процессе*/
    std::vector<double> fullG(static_cast<size_t>(padded_rows) * N, INF);

    /*собираем начальную матрицу на всех процессах*/
    MPI_Allgather(localG.data(), rows_per_proc * N, MPI_DOUBLE,
                  fullG.data(), rows_per_proc * N, MPI_DOUBLE,
                  MPI_COMM_WORLD);

    /*алгоритм Флойда–Уоршелла*/
    for (int k = 0; k < N; k++) {
        /*строка k должна быть доступна всем процессам*/
        const double* rowK = &fullG[static_cast<size_t>(k) * N];

        /*обновляем только реальные строки своего блока*/
        for (int i = 0; i < rows_per_proc; i++) {
            int global_i = rank * rows_per_proc + i;
            if (global_i >= N) continue;

            double dik = localG[static_cast<size_t>(i) * N + k];
            if (dik >= INF / 2) continue;

            for (int j = 0; j < N; j++) {
                double alt = dik + rowK[j];
                double& dij = localG[static_cast<size_t>(i) * N + j];
                if (alt < dij) dij = alt;
            }
        }

        /*обмен обновлёнными данными между процессами*/
        MPI_Allgather(localG.data(), rows_per_proc * N, MPI_DOUBLE,
                      fullG.data(), rows_per_proc * N, MPI_DOUBLE,
                      MPI_COMM_WORLD);
    }

    /*подготавливаем буфер для сборки итоговой матрицы на rank=0*/
    if (rank == 0) {
        G.assign(static_cast<size_t>(padded_rows) * N, INF);
    }

    /*собираем итоговую матрицу на процессе rank=0*/
    MPI_Gather(localG.data(), rows_per_proc * N, MPI_DOUBLE,
               rank == 0 ? G.data() : nullptr,
               rows_per_proc * N, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    /*конец замера времени*/
    double end_time = MPI_Wtime();

    /*вывод результата выполняется только на rank=0*/
    if (rank == 0) {
        std::cout << "N = " << N << "\n";
        std::cout << "Processes = " << size << "\n";
        std::cout << "rows_per_proc = " << rows_per_proc
                  << " (padded_rows = " << padded_rows << ")\n";
        std::cout << "Shortest paths matrix (INF shown as -1):\n";

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                double v = Mat(G, i, j, N);
                if (v >= INF / 2) std::cout << -1 << " ";
                else std::cout << v << " ";
            }
            std::cout << "\n";
        }

        std::cout << "Execution time: "
                  << (end_time - start_time) << " seconds\n";
    }

    /*завершение работы MPI*/
    MPI_Finalize();
    return 0;
}
