
#include <vector>
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cstdlib>
using namespace std;

static void matmul_serial(const vector<double>& A,
                          const vector<double>& B,
                          vector<double>& C, int N)
{
    for (int i = 0; i < N; ++i) {
        const double* arow = &A[i*N];
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < N; ++k)
                acc += arow[k] * B[k*N + j];
            C[i*N + j] = acc;
        }
    }
}

int main(int argc, char** argv) {
    int N = (argc >= 2 ? atoi(argv[1]) : 600);
    int iters = (argc >= 3 ? atoi(argv[2]) : 1);

    vector<double> A(N*N), B(N*N), C(N*N, 0.0);
    for (int i = 0; i < N*N; ++i) { A[i]=sin(0.001*i); B[i]=cos(0.001*i); }

    auto t0 = chrono::steady_clock::now();
    for (int t=0; t<iters; ++t) matmul_serial(A,B,C,N);
    double ms = chrono::duration<double, milli>(chrono::steady_clock::now()-t0).count();

    double checksum=0.0; for (int i=0;i<min(N*N,100);++i) checksum+=C[i];
    cout << "time_ms=" << ms << "\n";
    cout << "checksum=" << fixed << setprecision(6) << checksum << "\n";
    return 0;
}
