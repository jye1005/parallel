// src/main_lab.cpp
#include "../Imagelib.h"
#include "../CModel.h"
#include "../CLayer.h"
#include <chrono>
#include <iostream>
using namespace std;

#ifdef USE_OMP
  #ifdef _OPENMP
    #include <omp.h>  
  #endif
#endif


int main() {
    #ifdef USE_OMP
      #ifdef _OPENMP
        cout << "[RUN] OpenMP ENABLED (USE_OMP)\n";
        cout << "      OMP_MAX_THREADS? use env OMP_NUM_THREADS, "
            << "default=" << omp_get_max_threads() << "\n";

        // 실제 사용된 thread 수 확인 (병렬 블록 안에서)
        #pragma omp parallel
        {
            #pragma omp single
            {
                cout << "      [OMP] threads actually used = "
                    << omp_get_num_threads() << "\n";
            }
        }
      #else
        cout << "[RUN] USE_OMP defined but compiler has no _OPENMP. (체크 플래그/링크 확인)\n";
      #endif
  #else
    cout << "[RUN] SERIAL (OpenMP disabled)\n";
  #endif


    Model model;

    // ─ build model (너가 준 구성 유지: 9x9 1→64, 5x5 64→32, 5x5 32→1)
    model.add_layer(new Layer_Conv("Conv1", 9, 1, 64, LOAD_INIT,
        "model/weights_conv1_9x9x1x64.txt", "model/biases_conv1_64.txt"));
    model.add_layer(new Layer_ReLU("Relu1", 1, 64, 64));
    model.add_layer(new Layer_Conv("Conv2", 5, 64, 32, LOAD_INIT,
        "model/weights_conv2_5x5x64x32.txt", "model/biases_conv2_32.txt"));
    model.add_layer(new Layer_ReLU("Relu2", 1, 32, 32));
    model.add_layer(new Layer_Conv("Conv3", 5, 32, 1, LOAD_INIT,
        "model/weights_conv3_5x5x32x1.txt", "model/biases_conv3_1.txt"));

    // ─ run & measure
    auto t0 = chrono::steady_clock::now();
    model.test("baby_512x512_input.bmp", "baby_512x512_output_srcnn.bmp");
    double ms = chrono::duration<double, std::milli>(chrono::steady_clock::now()-t0).count();

    cout << "elapsed_ms=" << ms << "\n";

    // (원하면) 상세 정보 출력
    model.print_layer_info();
    model.print_tensor_info();

    #ifdef _WIN32
      system("PAUSE");
    #endif
    return 0;
}
