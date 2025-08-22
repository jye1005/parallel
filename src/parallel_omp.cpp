
#include <omp.h>
#include <iostream>
#include <vector>
#include <chrono>
using namespace std;

int main(int argc, char** argv) {
    int nThreads = omp_get_max_threads();
    cout << "OMP max threads = " << nThreads << "\n";
    
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nt  = omp_get_num_threads();
#pragma omp critical
        cout << "hello from thread " << tid << "/" << nt << "\n";
    }
    
    // 간단한 연산도 해봄
    int N = (argc>=2 ? atoi(argv[1]) : 20000000);
    vector<int> a(N, 1);
    long long sum = 0;
    auto t0 = chrono::steady_clock::now();
#pragma omp parallel for reduction(+:sum) schedule(static)
    for (int i=0;i<N;++i) sum += a[i];
    auto ms = chrono::duration<double, milli>(chrono::steady_clock::now()-t0).count();
    cout << "sum=" << sum << ", time_ms=" << ms << "\n";
    return 0;
}
