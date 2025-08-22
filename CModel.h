#pragma once
#include "CLayer.h"
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>

using std::string;
using std::vector;
using std::cout;
using std::endl;

class Model {
private:
    vector<Layer*>    layers;
    vector<Tensor3D*> tensors;

public:
    Model() {}
    void add_layer(Layer* layer) { layers.push_back(layer); }

    ~Model() {
        for (Layer* layer : layers)   delete layer;
        for (Tensor3D* t  : tensors)  delete t;
        layers.clear();
        tensors.clear();
    }

    void test(string filename_input, string filename_output) {
        int nH, nW;
        double **input_img_Y, **input_img_U, **input_img_V;
        ::byte* pLoadImage;

        read_image(filename_input, pLoadImage, input_img_Y, input_img_U, input_img_V, nH, nW);
        cout << "Reading (" << filename_input << ") is complete..." << endl;

        for (size_t i = 0; i < layers.size(); ++i) {
            Tensor3D* out = layers[i]->forward(tensors.back());
            tensors.push_back(out);
            cout << "Layer " << (i + 1) << " forward complete..." << endl;
        }

        Tensor3D* outY = tensors.back();
        save_image_place_center(filename_output, pLoadImage, outY,
                                input_img_Y, input_img_U, input_img_V,
                                nH, nW);
        cout << "Saving (" << filename_output << ") is complete..." << endl;

        std::free(pLoadImage);
        free_dmatrix2D(input_img_Y, nH, nW);
        free_dmatrix2D(input_img_U, nH, nW);
        free_dmatrix2D(input_img_V, nH, nW);
    }

    void read_image(const string filename,
                    ::byte*& pLoadImage,
                    double**& img_Y, double**& img_U, double**& img_V,
                    int& nH, int& nW)
    {
        LoadBmp(filename.c_str(), &pLoadImage, nH, nW);

        img_Y = dmatrix2D(nH, nW);
        img_U = dmatrix2D(nH, nW);
        img_V = dmatrix2D(nH, nW);

        convert1Dto2D(pLoadImage, img_Y, img_U, img_V, nH, nW);

        double*** inImage3D = dmatrix3D(nH, nW, 1);
        convert2Dto3D(img_Y, inImage3D, nH, nW);

        Tensor3D* temp = new Tensor3D(nH, nW, 1);
        temp->set_tensor(inImage3D);  
        tensors.push_back(temp);
    }

    void save_image(string filename,
                    ::byte*& pLoadImage,
                    Tensor3D*& tensor_Y,
                    double** img_U, double** img_V,
                    int nH, int nW)
    {
        double** img_Y = dmatrix2D(nH, nW);
        convert3Dto2D(tensor_Y->get_tensor(), img_Y, nH, nW);
        convert2Dto1D(img_Y, img_U, img_V, pLoadImage, nH, nW);
        SaveBmp(filename.c_str(), pLoadImage, nH, nW);
        free_dmatrix2D(img_Y, nH, nW);
    }
    void save_image_place_center(const string& filename,
                                 ::byte*& pLoadImage,
                                 Tensor3D*& tensor_Y_out,
                                 double** inY, double** U, double** V,
                                 int nH, int nW)
    {
        double** Yfull = dmatrix2D(nH, nW);
        for (int y = 0; y < nH; ++y)
            for (int x = 0; x < nW; ++x)
                Yfull[y][x] = inY[y][x];

        int outH, outW, outC;
        tensor_Y_out->get_info(outH, outW, outC);

        double*** src = tensor_Y_out->get_tensor();

        int dy = std::max(0, (nH - outH) / 2);
        int dx = std::max(0, (nW - outW) / 2);
        int copyH = std::min(outH, nH - dy);
        int copyW = std::min(outW, nW - dx);

        for (int y = 0; y < copyH; ++y)
            for (int x = 0; x < copyW; ++x)
                Yfull[y + dy][x + dx] = src[y][x][0];

        convert2Dto1D(Yfull, U, V, pLoadImage, nH, nW);
        SaveBmp(filename.c_str(), pLoadImage, nH, nW);
        free_dmatrix2D(Yfull, nH, nW);
    }

    void print_layer_info() const {
        cout << endl << "(Layer information)_____________" << endl;
        for (unsigned i = 0; i < layers.size(); i++) {
            cout << i + 1 << "-th layer: ";
            layers.at(i)->print();
        }
    }

    void print_tensor_info() const {
        cout << endl << "(Tensor information)_____________" << endl;
        for (unsigned i = 0; i < tensors.size(); i++) {
            cout << i + 1 << "-th tensor: ";
            tensors.at(i)->print();
        }
    }
};
