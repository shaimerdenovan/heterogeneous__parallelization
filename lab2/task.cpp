/*Задание
1. Реализация сортировок без параллелизма:
Напишите функции для сортировки пузырьком, выбором и вставкой
без использования OpenMP.
2. Параллельная реализация с использованием OpenMP:
Используйте директивы OpenMP для распараллеливания внешних
циклов. Протестируйте производительность каждой сортировки на
массивах разного размера (например, 1000, 10,000 и 100,000
элементов).
3. Сравнение производительности:
Измерьте время выполнения последовательных и параллельных
версий каждой сортировки, используя библиотеку <chrono>.
Сравните результаты и сделайте выводы.*/

/*подключаем библиотеку для ввода и вывода*/
#include <iostream>
/*библиотека для работы с динамическими массивами*/
#include <vector>
/*библиотека для генерации случайных чисел*/
#include <random>
/*библиотека для измерения времени выполнения*/
#include <chrono>
/*библиотека для параллельных вычислений OpenMP*/
#include <omp.h>

/* используем стандартное пространство имён */
using namespace std;

/*Последовательные сортировки*/

/*пузырьковая сортировка*/
/*поочередно сравниваем соседние элементы и меняем их местами*/
void bubbleSortSequential(vector<int>& numbers) {
    int n = numbers.size();
    /*проходим по массиву несколько раз*/
    for (int i = 0; i < n - 1; i++) {
        /*сравниваем соседние элементы*/
        for (int j = 0; j < n - i - 1; j++) {
            if (numbers[j] > numbers[j + 1]) {
                swap(numbers[j], numbers[j + 1]);
            }
        }
    }
}

/*сортировка выбором*/
/*находим минимальный элемент и ставим его на нужное место*/
void selectionSortSequential(vector<int>& numbers) {
    int n = numbers.size();
    /*проходим по массиву*/
    for (int i = 0; i < n - 1; i++) {
        /*считаем текущий элемент минимальным*/
        int minIdx = i;
        /*ищем минимальный элемент в оставшейся части массива*/
        for (int j = i + 1; j < n; j++) {
            if (numbers[j] < numbers[minIdx]) {
                minIdx = j;
            }
        }
        /*меняем местами элементы*/
        swap(numbers[i], numbers[minIdx]);
    }
}

/*сортировка вставками*/
/*вставляем каждый элемент на правильное место*/
void insertionSortSequential(vector<int>& numbers) {
    int n = numbers.size();
    /*начинаем со второго элемента*/
    for (int i = 1; i < n; i++) {
        int key = numbers[i];
        int j = i - 1;
        /*сдвигаем элементы вправо*/
        while (j >= 0 && numbers[j] > key) {
            numbers[j + 1] = numbers[j];
            j--;
        }
        /*вставляем элемент на нужное место*/
        numbers[j + 1] = key;
    }
}

/*Параллельные сортировки(OpenMP)*/

/*параллельная пузырьковая сортировка (odd-even)*/
void bubbleSortParallel(vector<int>& numbers) {
    int n = numbers.size();
    /*выполняем несколько фаз сортировки*/
    for (int phase = 0; phase < n; phase++) {
        /*четная фаза*/
        if (phase % 2 == 0) {
#pragma omp parallel for
            for (int i = 0; i < n - 1; i += 2) {
                if (numbers[i] > numbers[i + 1]) {
                    swap(numbers[i], numbers[i + 1]);
                }
            }
        } 
        /*нечетная фаза*/
        else {
#pragma omp parallel for
            for (int i = 1; i < n - 1; i += 2) {
                if (numbers[i] > numbers[i + 1]) {
                    swap(numbers[i], numbers[i + 1]);
                }
            }
        }
    }
}

/*параллельная сортировка выбором*/
void selectionSortParallel(vector<int>& numbers) {
    int n = numbers.size();
    for (int i = 0; i < n - 1; i++) {
        int minIdx = i;
/*параллельно ищем минимальный элемент*/
#pragma omp parallel for
        for (int j = i + 1; j < n; j++) {
#pragma omp critical
            {
                if (numbers[j] < numbers[minIdx]) {
                    minIdx = j;
                }
            }
        }
        /*меняем элементы местами*/
        swap(numbers[i], numbers[minIdx]);
    }
}

/*параллельная сортировка вставками*/
void insertionSortParallel(vector<int>& numbers) {
    int n = numbers.size();

#pragma omp parallel for
    for (int i = 1; i < n; i++) {
        int key = numbers[i];
        int j = i - 1;
        while (j >= 0 && numbers[j] > key) {
            numbers[j + 1] = numbers[j];
            j--;
        }
        numbers[j + 1] = key;
    }
}

/*функция генерации массива случайных чисел*/
vector<int> generateArray(int n) {
    vector<int> numbers(n);
    /*источник случайных чисел*/
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist(0, 100000);

    /*заполняем массив*/
    for (int i = 0; i < n; i++) {
        numbers[i] = dist(gen);
    }
    return numbers;
}

/*главная функция*/
int main() {
    /*размеры массивов для тестирования*/
    vector<int> sizes = {1000, 10000, 100000};

    /*перебираем размеры массивов*/
    for (int n : sizes) {
        cout << "\nArray size: " << n << endl;

        /*создаем исходный массив*/
        vector<int> base = generateArray(n);

        /*пузырьковая сортировка*/
        vector<int> numbers = base;

        /*последовательная версия*/
        auto start = chrono::high_resolution_clock::now();
        bubbleSortSequential(numbers);
        auto end = chrono::high_resolution_clock::now();
        double bubble_seq = chrono::duration<double>(end - start).count();

        /*параллельная версия*/
        numbers = base;
        start = chrono::high_resolution_clock::now();
        bubbleSortParallel(numbers);
        end = chrono::high_resolution_clock::now();
        double bubble_par = chrono::duration<double>(end - start).count();

        cout << "Bubble sort:\n";
        cout << "Sequential:" << bubble_seq << " c\n";
        cout << "Parallel:" << bubble_par << " c\n";

        /*сортировка выбором*/
        numbers = base;
        start = chrono::high_resolution_clock::now();
        selectionSortSequential(numbers);
        end = chrono::high_resolution_clock::now();
        double sel_seq = chrono::duration<double>(end - start).count();

        numbers = base;
        start = chrono::high_resolution_clock::now();
        selectionSortParallel(numbers);
        end = chrono::high_resolution_clock::now();
        double sel_par = chrono::duration<double>(end - start).count();

        cout << "\nSelection sort:\n";
        cout << "Sequential:" << sel_seq << " c\n";
        cout << "Parallel:" << sel_par << " c\n";

        /*сортировка вставками*/
        numbers = base;
        start = chrono::high_resolution_clock::now();
        insertionSortSequential(numbers);
        end = chrono::high_resolution_clock::now();
        double ins_seq = chrono::duration<double>(end - start).count();

        numbers = base;
        start = chrono::high_resolution_clock::now();
        insertionSortParallel(numbers);
        end = chrono::high_resolution_clock::now();
        double ins_par = chrono::duration<double>(end - start).count();

        cout << "\nInsertion sort:\n";
        cout << "Sequential:" << ins_seq << " c\n";
        cout << "Parallel:" << ins_par << " c\n";

    }

    /*завершение программы*/
    return 0;
}