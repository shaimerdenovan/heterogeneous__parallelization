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


/*используем OpenCL 1.2*/
#define CL_TARGET_OPENCL_VERSION 120

/*подключаем заголовок OpenCL*/
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

/*библиотека для измерения времени на CPU*/
#include <chrono>
/*библиотека для работы со строками*/
#include <cstring>
/*библиотека для чтения kernel.cl из файла*/
#include <fstream>
/*библиотека ввода/вывода*/
#include <iostream>
/*библиотека random для генерации случайных чисел*/
#include <random>
/*библиотека string*/
#include <string>
/*библиотека vector для хранения данных на CPU*/
#include <vector>
/*библиотека cmath для fabs*/
#include <cmath>


/*функция-обработчик ошибок OpenCL
если какая-то функция OpenCL вернула ошибку, печатаем сообщение и завершаем программу*/
static void die(const std::string& msg, cl_int err) {
    std::cerr << "ERROR: " << msg << " (cl_int=" << err << ")\n";
    std::exit(1);
}


/*чтение текста из файла (используем для kernel.cl)
возвращает содержимое файла как одну строку*/
static std::string readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Cannot open file: " << path << "\n";
        std::exit(1);
    }
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return s;
}


/*перевод типа устройства OpenCL в строку для вывода*/
static const char* deviceTypeToStr(cl_device_type t) {
    if (t & CL_DEVICE_TYPE_GPU) return "GPU";
    if (t & CL_DEVICE_TYPE_CPU) return "CPU";
    if (t & CL_DEVICE_TYPE_ACCELERATOR) return "ACCELERATOR";
    return "UNKNOWN";
}


/*вывод информации об устройстве (vendor/name/type)*/
static void printDeviceInfo(cl_device_id dev) {
    char name[256]{};
    char vendor[256]{};
    cl_device_type dtype{};
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    clGetDeviceInfo(dev, CL_DEVICE_VENDOR, sizeof(vendor), vendor, nullptr);
    clGetDeviceInfo(dev, CL_DEVICE_TYPE, sizeof(dtype), &dtype, nullptr);

    std::cout << "Selected device: " << vendor << " / " << name
              << " [" << deviceTypeToStr(dtype) << "]\n";
}


/*выбор устройства OpenCL по параметру --device cpu|gpu
получаем список платформ,пытаемся найти устройство нужного типа (CPU или GPU),если не нашли то берём первое доступное устройство*/
static cl_device_id pickDevice(const std::string& want) {
    cl_int err;

    /*получаем число доступных платформ OpenCL*/
    cl_uint numPlatforms = 0;
    err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) die("clGetPlatformIDs", err);

    /*получаем список платформ*/
    std::vector<cl_platform_id> platforms(numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    if (err != CL_SUCCESS) die("clGetPlatformIDs(list)", err);

    /*какой тип устройства пользователь хочет*/
    cl_device_type preferred = 0;
    if (want == "gpu") preferred = CL_DEVICE_TYPE_GPU;
    else if (want == "cpu") preferred = CL_DEVICE_TYPE_CPU;
    else {
        std::cerr << "--device must be cpu or gpu\n";
        std::exit(1);
    }

    /*1.пытаемся найти устройство нужного типа*/
    for (auto p : platforms) {
        cl_uint ndev = 0;
        err = clGetDeviceIDs(p, preferred, 0, nullptr, &ndev);
        if (err == CL_SUCCESS && ndev > 0) {
            std::vector<cl_device_id> devs(ndev);
            err = clGetDeviceIDs(p, preferred, ndev, devs.data(), nullptr);
            if (err != CL_SUCCESS) die("clGetDeviceIDs(preferred)", err);
            return devs[0]; /*берём первое устройство*/
        }
    }

    /*2.fallback: если нужного типа нет то берём любое доступное устройство*/
    for (auto p : platforms) {
        cl_uint ndev = 0;
        err = clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, nullptr, &ndev);
        if (err == CL_SUCCESS && ndev > 0) {
            std::vector<cl_device_id> devs(ndev);
            err = clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, ndev, devs.data(), nullptr);
            if (err != CL_SUCCESS) die("clGetDeviceIDs(fallback)", err);
            std::cerr << "WARNING: Preferred device type not found, using fallback device.\n";
            return devs[0];
        }
    }

    std::cerr << "No OpenCL devices found.\n";
    std::exit(1);
}


/*создание и компиляция OpenCL программы из исходного текста kernel.cl
если компиляция не прошла то выводим build log*/
static cl_program buildProgram(cl_context ctx, cl_device_id dev, const std::string& src) {
    cl_int err;

    const char* s = src.c_str();
    size_t len = src.size();

    /*создаём объект программы из текста*/
    cl_program prog = clCreateProgramWithSource(ctx, 1, &s, &len, &err);
    if (err != CL_SUCCESS) die("clCreateProgramWithSource", err);

    /*компилируем программу под выбранное устройство*/
    err = clBuildProgram(prog, 1, &dev, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        /*если ошибка то печатаем лог компиляции*/
        size_t logSize = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::string log(logSize, '\0');
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        std::cerr << "Build log:\n" << log << "\n";
        die("clBuildProgram", err);
    }
    return prog;
}


/*получаем время выполнения команды из cl_event
OpenCL возвращает START и END в наносекундах и переводим в миллисекунды*/
static double eventElapsedMs(cl_event ev) {
    cl_ulong start = 0, end = 0;
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr);
    return (double)(end - start) * 1e-6; /*ns -> ms*/
}


/*генерация случайных чисел в массиве float на CPU (диапазон [-1;1])*/
static void fillRandom(std::vector<float>& v, unsigned seed = 123) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(rng);
}


/*проверка приблизительного равенства float
eps - относительная погрешность*/
static bool almostEqual(float a, float b, float eps = 1e-3f) {
    float diff = std::fabs(a - b);
    float norm = std::max(1.0f, std::max(std::fabs(a), std::fabs(b)));
    return diff <= eps * norm;
}


/*Задача №1: Векторное сложение
Здесь мы генерируем A и B на CPU, считаем reference на CPU,создаём буферы OpenCL и копируем A,B,
запускаем ядро vector_add,считываем C и сравниваем с reference, печатаем время CPU и время ядра OpenCL*/
static void runVectorAdd(cl_context ctx, cl_command_queue q, cl_program prog, int n) {
    cl_int err;

    /*создаём массивы на CPU*/
    std::vector<float> A(n), B(n), C(n), ref(n);

    /*заполняем случайными значениями*/
    fillRandom(A, 1);
    fillRandom(B, 2);

    /*CPU reference+замер времени*/
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++) ref[i] = A[i] + B[i];
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    /*создаём OpenCL буферы в global memory устройства CL_MEM_COPY_HOST_PTR сразу копирует данные из host памяти*/
    cl_mem bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * (size_t)n, A.data(), &err);
    if (err != CL_SUCCESS) die("clCreateBuffer(A)", err);

    cl_mem bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * (size_t)n, B.data(), &err);
    if (err != CL_SUCCESS) die("clCreateBuffer(B)", err);

    /*буфер результата C создаём пустой*/
    cl_mem bufC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                 sizeof(float) * (size_t)n, nullptr, &err);
    if (err != CL_SUCCESS) die("clCreateBuffer(C)", err);

    /*создаём объект ядра vector_add из программы*/
    cl_kernel k = clCreateKernel(prog, "vector_add", &err);
    if (err != CL_SUCCESS) die("clCreateKernel(vector_add)", err);

    /*передаём аргументы ядру 0: A, 1: B, 2: C, 3: n*/
    err = clSetKernelArg(k, 0, sizeof(cl_mem), &bufA);
    err |= clSetKernelArg(k, 1, sizeof(cl_mem), &bufB);
    err |= clSetKernelArg(k, 2, sizeof(cl_mem), &bufC);
    err |= clSetKernelArg(k, 3, sizeof(int), &n);
    if (err != CL_SUCCESS) die("clSetKernelArg(vector_add)", err);

    /*размер глобальной области: n work-item (по одному на элемент)*/
    size_t global = (size_t)n;

    /*запускаем ядро; сохраняем событие ev для профилирования*/
    cl_event ev{};
    err = clEnqueueNDRangeKernel(q, k, 1, nullptr, &global, nullptr, 0, nullptr, &ev);
    if (err != CL_SUCCESS) die("clEnqueueNDRangeKernel(vector_add)", err);

    /*ждём завершения ядра*/
    err = clWaitForEvents(1, &ev);
    if (err != CL_SUCCESS) die("clWaitForEvents(vector_add)", err);

    /*время работы только ядра (без чтения результата)*/
    double kernel_ms = eventElapsedMs(ev);

    /*читаем буфер результата обратно на CPU*/
    err = clEnqueueReadBuffer(q, bufC, CL_TRUE, 0, sizeof(float) * (size_t)n, C.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) die("clEnqueueReadBuffer(C)", err);

    /*проверяем корректность*/
    int bad = 0;
    for (int i = 0; i < n; i++) {
        if (!almostEqual(C[i], ref[i])) {
            bad++;
            if (bad < 5) std::cerr << "Mismatch at " << i << "\n";
        }
    }

    /*печать результатов*/
    std::cout << "\n=== Vector Add ===\n";
    std::cout << "n = " << n << "\n";
    std::cout << "CPU reference time: " << cpu_ms << " ms (single-thread loop)\n";
    std::cout << "OpenCL kernel time: " << kernel_ms << " ms (device)\n";
    std::cout << "Correctness: " << (bad == 0 ? "OK" : "FAIL, bad=" + std::to_string(bad)) << "\n";

    /*освобождаем ресурсы OpenCL*/
    clReleaseEvent(ev);
    clReleaseKernel(k);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
}


/*Задача №2: Умножение матриц
Здесь мы генерируем A и B на CPU,считаем reference на CPU (3 цикла),создаём буферы OpenCL и копируем A,B,
запускаем ядро mat_mul в 2D (global = (K, N)),считываем C и сравниваем с reference*/
static void runMatMul(cl_context ctx, cl_command_queue q, cl_program prog, int N, int M, int K) {
    cl_int err;

    /*матрицы на CPU: A: N×M, B: M×K, C: N×K*/
    std::vector<float> A((size_t)N * M), B((size_t)M * K), C((size_t)N * K), ref((size_t)N * K);

    /*заполняем случайными значениями*/
    fillRandom(A, 3);
    fillRandom(B, 4);

    /*CPU reference+замер времени*/
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < K; c++) {
            float sum = 0.0f;
            for (int i = 0; i < M; i++) {
                sum += A[(size_t)r * M + i] * B[(size_t)i * K + c];
            }
            ref[(size_t)r * K + c] = sum;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    /*создаём буферы OpenCL и копируем A,B на устройство*/
    cl_mem bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * (size_t)A.size(), A.data(), &err);
    if (err != CL_SUCCESS) die("clCreateBuffer(A)", err);

    cl_mem bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * (size_t)B.size(), B.data(), &err);
    if (err != CL_SUCCESS) die("clCreateBuffer(B)", err);

    cl_mem bufC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                 sizeof(float) * (size_t)C.size(), nullptr, &err);
    if (err != CL_SUCCESS) die("clCreateBuffer(C)", err);

    /*создаём ядро mat_mul*/
    cl_kernel k = clCreateKernel(prog, "mat_mul", &err);
    if (err != CL_SUCCESS) die("clCreateKernel(mat_mul)", err);

    /*передаём аргументы ядру A,B,C,N,M,K*/
    err = clSetKernelArg(k, 0, sizeof(cl_mem), &bufA);
    err |= clSetKernelArg(k, 1, sizeof(cl_mem), &bufB);
    err |= clSetKernelArg(k, 2, sizeof(cl_mem), &bufC);
    err |= clSetKernelArg(k, 3, sizeof(int), &N);
    err |= clSetKernelArg(k, 4, sizeof(int), &M);
    err |= clSetKernelArg(k, 5, sizeof(int), &K);
    if (err != CL_SUCCESS) die("clSetKernelArg(mat_mul)", err);

    /*2D запуск:
    get_global_id(0) - col (0..K-1)
    get_global_id(1) - row (0..N-1)*/
    size_t global[2] = {(size_t)K, (size_t)N};

    /*local size можно подобрать*/
    cl_event ev{};
    err = clEnqueueNDRangeKernel(q, k, 2, nullptr, global, nullptr, 0, nullptr, &ev);
    if (err != CL_SUCCESS) die("clEnqueueNDRangeKernel(mat_mul)", err);

    /*ждём завершения ядра*/
    err = clWaitForEvents(1, &ev);
    if (err != CL_SUCCESS) die("clWaitForEvents(mat_mul)", err);

    /*время работы только ядра*/
    double kernel_ms = eventElapsedMs(ev);

    /*читаем результат обратно на CPU*/
    err = clEnqueueReadBuffer(q, bufC, CL_TRUE, 0, sizeof(float) * (size_t)C.size(), C.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) die("clEnqueueReadBuffer(C)", err);

    /*проверка корректности*/
    int bad = 0;
    for (size_t i = 0; i < C.size(); i++) {
        if (!almostEqual(C[i], ref[i], 1e-2f)) {
            bad++;
            if (bad < 5) std::cerr << "Mismatch at " << i << "\n";
        }
    }

    /*печать результатов*/
    std::cout << "\n=== Matrix Multiply ===\n";
    std::cout << "A: " << N << "x" << M << ", B: " << M << "x" << K << ", C: " << N << "x" << K << "\n";
    std::cout << "CPU reference time: " << cpu_ms << " ms (triple loop)\n";
    std::cout << "OpenCL kernel time: " << kernel_ms << " ms (device)\n";
    std::cout << "Correctness: " << (bad == 0 ? "OK" : "FAIL, bad=" + std::to_string(bad)) << "\n";

    /*освобождаем ресурсы*/
    clReleaseEvent(ev);
    clReleaseKernel(k);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
}


/*чтение целочисленного аргумента командной строки вида --n 10000000
если аргумент не найден, возвращаем значение по умолчанию def*/
static int getArgInt(int argc, char** argv, const std::string& name, int def) {
    for (int i = 1; i < argc - 1; i++) {
        if (name == argv[i]) return std::stoi(argv[i + 1]);
    }
    return def;
}


/*чтение строкового аргумента командной строки вида --device cpu
если аргумент не найден, возвращаем значение по умолчанию def*/
static std::string getArgStr(int argc, char** argv, const std::string& name, const std::string& def) {
    for (int i = 1; i < argc - 1; i++) {
        if (name == argv[i]) return argv[i + 1];
    }
    return def;
}


/*функция main()
Здесь мы читаем аргументы командной строки, выбираем устройство OpenCL (CPU/GPU),создаём контекст и командную очередь (с профилированием),
читаем kernel.cl и компилируем программу,запускаем выбранную задачу (vec или mat),освобождаем ресурсы*/
int main(int argc, char** argv) {
    /*по умолчанию просим GPU, но если GPU нет то pickDevice сделает fallback*/
    std::string deviceWant = getArgStr(argc, argv, "--device", "gpu"); /*cpu|gpu*/
    std::string task       = getArgStr(argc, argv, "--task", "vec");  /*vec|mat*/

    /*параметры задачи vec*/
    int n = getArgInt(argc, argv, "--n", 1 << 24);

    /*параметры задачи mat*/
    int N = getArgInt(argc, argv, "--N", 512);
    int M = getArgInt(argc, argv, "--M", 512);
    int K = getArgInt(argc, argv, "--K", 512);

    cl_int err;

    /*выбор устройства (CPU или GPU)*/
    cl_device_id dev = pickDevice(deviceWant);
    printDeviceInfo(dev);

    /*создаём контекст OpenCL*/
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) die("clCreateContext", err);

    /*создаём командную очередь CL_QUEUE_PROFILING_ENABLE нужен для event profiling (время ядра)*/
    cl_command_queue q = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) die("clCreateCommandQueue", err);

    /*читаем исходный код ядра и компилируем*/
    std::string src = readFile("kernel.cl");
    cl_program prog = buildProgram(ctx, dev, src);

    /*запуск выбранной задачи*/
    if (task == "vec") {
        runVectorAdd(ctx, q, prog, n);
    } else if (task == "mat") {
        runMatMul(ctx, q, prog, N, M, K);
    } else {
        std::cerr << "--task must be vec or mat\n";
    }

    /*освобождаем ресурсы OpenCL*/
    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);

    /*завершение программы*/
    return 0;
}
