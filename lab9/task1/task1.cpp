/*Задание 1: Распределённое вычисление среднего значения и стандартного отклонения 
Описание задачи: 
Напишите программу, которая выполняет следующие шаги: 
1. Создайте массив случайных чисел на процессе с "rank = 0". Размер массива — N (например, N = 10^6). 
2. Разделите массив между всеми процессами с помощью функции "MPI_Scatterv" (учитывая, что массив может не делиться нацело между процессами). 
3. Каждый процесс вычисляет: - Сумму элементов своей части массива. - Сумму квадратов элементов своей части массива. 
4. Соберите локальные суммы на процессе с "rank = 0" с помощью функции "MPI_Reduce". 
5. На основе собранных данных вычислите: - Среднее значение массива. - Стандартное отклонение, формула:
6. Выведите результаты на экран.*/

/*библиотека MPI: инициализация, обмен данными, замер времени*/
#include <mpi.h>
/*стандартный ввод/вывод*/
#include <iostream>
/*контейнер vector для хранения массивов*/
#include <vector>
/*генерация случайных чисел*/
#include <random>
/*математические функции (sqrt)*/
#include <cmath>
/*вспомогательные численные операции*/
#include <numeric>

/*главная функция программы*/
int main(int argc, char** argv) {
    /*инициализация MPI*/
    MPI_Init(&argc, &argv);

    /*rank-номер текущего процесса, size-общее количество процессов*/
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /*размер массива N*/
    long long N = 1000000;
    if (argc >= 2) N = std::stoll(argv[1]);

    /*замер времени начала выполнения*/
    double start_time = MPI_Wtime();

    /*массив данных создаётся только на процессе rank=0*/
    std::vector<double> data;

    /*массивы counts и displs нужны для MPI_Scatterv: counts-сколько элементов получает каждый процесс,
    displs-с какого индекса начинается его часть */
    std::vector<int> counts(size), displs(size);

    /*вычисляем базовое количество элементов на процесс и остаток при делении*/
    long long base = N / size;
    int rem = static_cast<int>(N % size);

    /*распределяем элементы между процессами, первые rem процессов получают на 1 элемент больше*/
    for (int i = 0; i < size; i++) {
        counts[i] = static_cast<int>(base + (i < rem ? 1 : 0));
    }

    /*вычисляем смещения в исходном массиве*/
    displs[0] = 0;
    for (int i = 1; i < size; i++) {
        displs[i] = displs[i - 1] + counts[i - 1];
    }

    /*процесс rank=0 заполняет массив случайными числами*/
    if (rank == 0) {
        data.resize(static_cast<size_t>(N));

        /*генератор случайных чисел*/
        std::mt19937_64 gen(12345);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        /*заполнение массива*/
        for (long long i = 0; i < N; i++) {
            data[static_cast<size_t>(i)] = dist(gen);
        }
    }

    /*локальный массив для каждого процесса*/
    int local_n = counts[rank];
    std::vector<double> local(static_cast<size_t>(local_n), 0.0);

    /*распределяем массив между процессами с учётом остатка*/
    MPI_Scatterv(
        rank == 0 ? data.data() : nullptr,
        counts.data(),
        displs.data(),
        MPI_DOUBLE,
        local.data(),
        local_n,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    /*локальные суммы: сумма элементов и сумма квадратов элементов*/
    double local_sum = 0.0;
    double local_sq_sum = 0.0;

    /*каждый процесс обрабатывает только свою часть массива*/
    for (double x : local) {
        local_sum += x;
        local_sq_sum += x * x;
    }

    /*глобальные суммы (будут собраны на rank=0)*/
    double total_sum = 0.0;
    double total_sq_sum = 0.0;

    /*собираем результаты со всех процессов*/
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sq_sum, &total_sq_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    /*замер времени окончания выполнения*/
    double end_time = MPI_Wtime();

    /*вычисления выполняются только на процессе rank=0*/
    if (rank == 0) {
        /*вычисление среднего значения*/
        double mean = total_sum / static_cast<double>(N);

        /*вычисление стандартного отклонения по формуле из задания*/
        double ex2 = total_sq_sum / static_cast<double>(N);
        double sigma = std::sqrt(ex2 - mean * mean);

        /*вывод результатов*/
        std::cout << "N = " << N << "\n";
        std::cout << "Processes = " << size << "\n";
        std::cout << "Mean = " << mean << "\n";
        std::cout << "Std (sigma) = " << sigma << "\n";
        std::cout << "Execution time: "
                  << (end_time - start_time) << " seconds\n";
    }

    /*завершение работы MPI*/
    MPI_Finalize();
    return 0;
}
