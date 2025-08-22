//
//  CTensor.h
//  parallel
//
//  Created by jihye on 8/19/25.
//

#pragma once
#include "Imagelib.h"
#include <iostream>
#include <iomanip> 

using namespace std;

class Tensor3D {
private:
    double*** tensor;
    int nH;
    int nW;
    int nC;
public:
    Tensor3D(int _nH, int _nW, int _nC): nH(_nH), nW(_nW), nC(_nC) {
        tensor = dmatrix3D(_nH, _nW, _nC);
        
        for (int c = 0; c < nC; c++){
            for (int w =0; w < nW; w++){
                for (int h=0; h <nH; h++){
                    tensor[h][w][c] = 0.0;
                }
            }
        }
    }
    ~Tensor3D() { free_dmatrix3D(tensor, nH, nW, nC);}
    void set_elem(int _h, int _w, int _c, double _val){ tensor[_h][_w][_c] = _val;}
    double get_elem(int _h, int _w, int _c) const { return tensor[_h][_w][_c];}
    
    void get_info(int& _nH, int& _nW, int& _nC) const {
        _nH = nH;
        _nW = nW;
        _nC = nC;}
    
    void set_tensor(double*** _tensor) { tensor = _tensor ; }
    double*** get_tensor() const {return tensor;}
    
    void print() const { std::cout<<nH<< "*" <<nW<< "*" <<nC << endl;}
};
