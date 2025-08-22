#pragma once
#include "Imagelib.h"
#include "CTensor.h"

#include <string>
#include <iostream>
#include <fstream>
#include <cassert>
#include <algorithm>

#ifdef USE_OMP
  #include <omp.h>
#endif

#define MEAN_INIT 0
#define LOAD_INIT 1

class Layer {
protected:
    int fK;
    int fC_in;
    int fC_out;
    std::string name;
public:
    Layer(std::string _name, int _fK, int _fC_in, int _fC_out);
    virtual ~Layer();
    virtual Tensor3D* forward(const Tensor3D* input) = 0;
    virtual void print() const = 0;
    virtual void get_info(std::string& _name, int& _fK, int& _fC_in, int& _fC_out) const = 0;
};

class Layer_ReLU : public Layer {
public:
    Layer_ReLU(std::string _name, int _fK, int _fC_in, int _fC_out);
    ~Layer_ReLU();
    Tensor3D* forward(const Tensor3D* input) override;
    void get_info(std::string& _name, int& _fK, int& _fC_in, int& _fC_out) const override;
    void print() const override;
};

class Layer_Conv : public Layer {
private:
    std::string filename_weight;
    std::string filename_bias;
    double****  weight_tensor;   
    double*     bias_tensor;    
    bool        weights_loaded {false};
public:
    Layer_Conv(std::string _name, int _fK, int _fC_in, int _fC_out,
               int init_type, std::string _filename_weight = "", std::string _filename_bias = "");
    ~Layer_Conv() override;
    void init(int init_type);
    Tensor3D* forward(const Tensor3D* input) override;  
    void get_info(std::string& _name, int& _fK, int& _fC_in, int& _fC_out) const override;
    void print() const override;
};

inline Layer::Layer(std::string _name, int _fK, int _fC_in, int _fC_out)
: fK(_fK), fC_in(_fC_in), fC_out(_fC_out), name(std::move(_name)) {}

inline Layer::~Layer() = default;

// ---- ReLU (1x1, 채널/공간 변경 없음) ----
inline Layer_ReLU::Layer_ReLU(std::string _name, int _fK, int _fC_in, int _fC_out)
: Layer(std::move(_name), _fK, _fC_in, _fC_out) {}

inline Layer_ReLU::~Layer_ReLU() = default;

inline Tensor3D* Layer_ReLU::forward(const Tensor3D* input) {
    int nH, nW, nC;
    input->get_info(nH, nW, nC);
    Tensor3D* out = new Tensor3D(nH, nW, nC);
    for (int c = 0; c < nC; ++c)
        for (int h = 0; h < nH; ++h)
            for (int w = 0; w < nW; ++w) {
                const double v = input->get_elem(h, w, c);
                out->set_elem(h, w, c, v > 0.0 ? v : 0.0);
            }
    return out;
}

inline void Layer_ReLU::get_info(std::string& _name, int& _fK, int& _fC_in, int& _fC_out) const {
    _name = name; _fK = fK; _fC_in = fC_in; _fC_out = fC_out;
}

inline void Layer_ReLU::print() const {
    std::cout << "[ReLU] name=" << name
              << " fK=" << fK << " C_in=" << fC_in << " C_out=" << fC_out << std::endl;
}

inline Layer_Conv::Layer_Conv(std::string _name, int _fK, int _fC_in, int _fC_out,
                              int init_type, std::string _filename_weight, std::string _filename_bias)
: Layer(std::move(_name), _fK, _fC_in, _fC_out),
  filename_weight(std::move(_filename_weight)),
  filename_bias(std::move(_filename_bias)),
  weight_tensor(nullptr),
  bias_tensor(nullptr)
{
    weight_tensor = dmatrix4D(fK, fK, fC_in, fC_out);
    bias_tensor   = new double[fC_out]();
    init(init_type);
}

inline Layer_Conv::~Layer_Conv() {
    if (weight_tensor) free_dmatrix4D(weight_tensor, fK, fK, fC_in, fC_out);
    delete[] bias_tensor;
}

inline void Layer_Conv::init(int init_type) {
    const double mean = 1.0 / std::max(1, fK * fK * fC_in);

    if (init_type == LOAD_INIT && !filename_weight.empty() && !filename_bias.empty()) {
        std::ifstream win(filename_weight);
        std::ifstream bin(filename_bias);
        if (win && bin) {
            for (int co = 0; co < fC_out; ++co)
                for (int ci = 0; ci < fC_in; ++ci)
                    for (int kh = 0; kh < fK; ++kh)
                        for (int kw = 0; kw < fK; ++kw)
                            win >> weight_tensor[kh][kw][ci][co];
            for (int co = 0; co < fC_out; ++co) bin >> bias_tensor[co];
            std::cerr << "[Conv:" << name << "] loaded weights/bias.\n";
            weights_loaded = true;
            return;
        } else {
            std::cerr << "[Conv:" << name << "] FAILED to load weights/bias. (fallback to MEAN_INIT)\n";
        }
    }

    for (int co = 0; co < fC_out; ++co)
        for (int ci = 0; ci < fC_in; ++ci)
            for (int kh = 0; kh < fK; ++kh)
                for (int kw = 0; kw < fK; ++kw)
                    weight_tensor[kh][kw][ci][co] = mean;
    std::fill(bias_tensor, bias_tensor + fC_out, 0.0);
    weights_loaded = false;
}

inline Tensor3D* Layer_Conv::forward(const Tensor3D* input) {
    int inH, inW, inC;
    input->get_info(inH, inW, inC);

    const int outH = inH - fK + 1;
    const int outW = inW - fK + 1;
    assert(outH > 0 && outW > 0 && "Input too small for kernel");

    Tensor3D* out = new Tensor3D(outH, outW, fC_out);

    #ifdef USE_OMP
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
    for (int co = 0; co < fC_out; ++co) {
        for (int y = 0; y < outH; ++y) {
            for (int x = 0; x < outW; ++x) {
                double acc = bias_tensor[co];
                for (int ci = 0; ci < fC_in; ++ci) {
                    for (int kh = 0; kh < fK; ++kh) {
                        for (int kw = 0; kw < fK; ++kw) {
                            const double v = (ci < inC) ? input->get_elem(y + kh, x + kw, ci) : 0.0;
                            acc += v * weight_tensor[kh][kw][ci][co];
                        }
                    }
                }
                out->set_elem(y, x, co, acc);
            }
        }
    }
    return out;
}
inline void Layer_Conv::get_info(std::string& _name, int& _fK, int& _fC_in, int& _fC_out) const {
    _name = name; _fK = fK; _fC_in = fC_in; _fC_out = fC_out;
}

inline void Layer_Conv::print() const {
    std::cout << "[Conv] name=" << name
              << " k=" << fK << " Cin=" << fC_in << " Cout=" << fC_out << std::endl;
}
