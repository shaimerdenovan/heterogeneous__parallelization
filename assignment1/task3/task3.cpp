/*Задание 3 
Используя OpenMP, реализуйте параллельный поиск минимального и максимального
элементов массива из задания 2. Сравните время выполнения последовательной и
параллельной реализаций.*/

/*подключаем библиотеку для ввода и вывода данных*/
#include <iostream>        
/*библиотека для генерации случайных чисел*/
#include <random>
/*библиотека для измерения времени выполнения*/
#include <chrono>
/*библиотека для получения минимальных и максимальных значений*/
#include <limits>
/*библиотека для параллельных вычислений OpenMP*/
#include <omp.h>

/*используем стандартное пространство имён чтоб не писать std:: каждый раз*/
using namespace std;

/*главная функция*/
int main() {

    /*размер массива устанавливаем 1000000 элементов*/
    const int SIZE = 1000000;
    /*динамически выделяем память под массив*/
    int* array = new int[SIZE];

    /*создаем источник случайных чисел*/
    random_device rd;
    /*генератор случайных чисел*/
    mt19937 gen(rd());
    /*задаем диапазон от 1 до 100*/
    uniform_int_distribution<int> dist(1, 100);

    /*заполняем массив случайными числами*/
    for (int i = 0; i < SIZE; i++) {
        array[i] = dist(gen);
    }

    /*переменные для хранения общего минимума и максимума*/
    int global_minimum = numeric_limits<int>::max();
    int global_maximum = numeric_limits<int>::min();

    /*запоминаем время начала параллельного поиска*/
    auto start = chrono::high_resolution_clock::now();

    /*параллельная область OpenMP*/
    #pragma omp parallel
    {
        /*локальные минимум и максимум для каждого потока*/
        int local_minimum = numeric_limits<int>::max();
        int local_maximum = numeric_limits<int>::min();

        /*распределяем цикл между потоками*/
        #pragma omp for nowait
        for (int i = 0; i < SIZE; i++) {
            if (array[i] < local_minimum) {
                local_minimum = array[i];
            }
            if (array[i] > local_maximum) {
                local_maximum = array[i];
            }
        }

        /*объединяем результаты всех потоков*/
        #pragma omp critical
        {
            if (local_minimum < global_minimum) {
                global_minimum = local_minimum;
            }
            if (local_maximum > global_maximum) {
                global_maximum = local_maximum;
            }
        }
    }

    /*запоминаем время окончания выполнения*/
    auto end = chrono::high_resolution_clock::now();

    /*вычисляем затраченное время в секундах*/
    chrono::duration<double> elapsed = end - start;

    /*выводим минимальное значение*/
    cout << "The minimum value is: " << global_minimum << endl;
    /*выводим максимальное значение*/
    cout << "The maximum value is: " << global_maximum << endl;
    /*выводим время выполнения программы*/
    cout << "The time in seconds is: " << elapsed.count() << endl;

    /*освобождаем динамически выделенную память*/
    delete[] array;

    /*обнуляем указатель*/
    array = nullptr;

    /*завершение программы */
    return 0;
}