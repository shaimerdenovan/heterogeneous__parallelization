/*lab3.cpp*/
/*Часть 3: Динамическая память и указатели
1. Реализуйте программу, которая создаёт динамический массив с помощью
указателей и заполняет его случайными числами.
2. Напишите функцию для поиска среднего значения элементов массива.
3. Параллельный подсчёт среднего значения
Используйте OpenMP для параллельного подсчёта суммы элементов и
вычисления среднего значения.
a. Добавьте директиву #pragma omp parallel for reduction(+:sum) для
параллельного суммирования элементов массива.
4. Освободите память после завершения работы с массивом.*/

/*подключаем библиотеку для ввода и вывода*/
#include <iostream>
/*библиотека для генерации случайных чисел*/
#include <random>
/*библиотека для замера времени*/
#include <chrono>

/*подключаем OpenMP*/
#ifdef _OPENMP
#include <omp.h>
#endif

int main() {
    /*используем пространство имён std, чтобы не писать std:: каждый раз*/
    using namespace std;

    /*ввод размера массива*/
    cout << "Enter N (array size): ";
    /*создаём переменную N*/
    size_t N;
    /*читаем N и проверяем, число должно быть введено корректно и не равно 0*/
    if (!(cin >> N) || N == 0) {
        /*если ввод неправильный то выводим сообщение об ошибке*/
        cerr << "Invalid N\n";
        return 1;
    }

    /*создаем динамический массив через указатель*/
    int* arr = new (nothrow) int[N];
    /*проверяем что память выделилась успешно*/
    if (!arr) {
        /*если память не выделилась то сообщаем об этом*/
        cerr << "Memory allocation failed\n";
        return 1;
    }

    /*на этом этапе задаем минимальное и максимальное числа в массиве*/
    const int RAND_MIN_VAL = 1;
    const int RAND_MAX_VAL = 100;

    /*это этап настройки генератора случайных чисел*/
    random_device rd;
    mt19937 gen(rd());
    /*делаем распределение чтоб получать числа в диапазоне 1 и 100*/
    uniform_int_distribution<int> dist(RAND_MIN_VAL, RAND_MAX_VAL);

    /*заполняем массив случайными числами*/
    for (size_t i = 0; i < N; ++i) {
        arr[i] = dist(gen);
    }

    /*если массив небольшой то выводим для проверки его целиком*/
    if (N <= 20) {
        cout << "Array elements: ";
        for (size_t i = 0; i < N; ++i) {
            cout << arr[i] << ' ';
        }
        cout << "\n";
    }

    /*этап последовательного подсчёта среднего значения*/
    /*создаём переменную для суммы при последовательном подсчёте*/
    double sum_seq = 0.0;

    /*запоминаем время старта последовательного подсчета*/
    auto t1 = chrono::high_resolution_clock::now();
    /*последовательно суммируем все элементы массива*/
    for (size_t i = 0; i < N; ++i) {
        sum_seq += arr[i];
    }
    /*запоминаем время окончания последовательного подсчета*/
    auto t2 = chrono::high_resolution_clock::now();

    /*считаем среднее значение*/
    double avg_seq = sum_seq / static_cast<double>(N);
    /*здесь считаем сколько времени занял расчет*/
    chrono::duration<double, milli> dur_seq = t2 - t1;

    /*это этап параллельного подсчета среднего значения с OpenMP*/
    /*готовим переменную для параллельной суммы*/
    double sum_par = 0.0;

    /*запоминаем время старта параллельного подсчета*/
    auto t3 = chrono::high_resolution_clock::now();
    /*если OpenMP доступен запускаем параллельное суммирование*/
#ifdef _OPENMP
    /*делим цикл между потоками и суммируем через reduction(+:sum_par)*/
    #pragma omp parallel for reduction(+:sum_par)
    for (int i = 0; i < static_cast<int>(N); ++i) {
        sum_par += arr[i];
    }
#else
    /*если OpenMP не включён при компиляции, считаем просто последовательно*/
    for (size_t i = 0; i < N; ++i) {
        sum_par += arr[i];
    }
#endif
    /*запоминаем время окончания параллельного подсчёта*/
    auto t4 = chrono::high_resolution_clock::now();

    /*считаем пареллельное среднее значение*/
    double avg_par = sum_par / static_cast<double>(N);
    /*считаем сколько времени занял параллельный расчет*/
    chrono::duration<double, milli> dur_par = t4 - t3;

    /*выводим результаты*/
    cout << "\nRESULTS:\n";
    cout << "Sequential average: " << avg_seq
         << " (time = " << dur_seq.count() << " ms)\n";
    cout << "Parallel   average: " << avg_par
         << " (time = " << dur_par.count() << " ms)\n";

/*выводим информацию про OpenMP*/
#ifdef _OPENMP
    /*выводим максимальное число потоков которое может использовать OpenMP*/
    cout << "OpenMP is enabled. Max threads: " << omp_get_max_threads() << "\n";
#else
    /*случай если OpenMP не включен*/
    cout << "OpenMP is NOT enabled (compile with -fopenmp to use it).\n";
#endif

    /*освобождаем память которую выделяли через new[]*/
    delete[] arr; 
    return 0;
}