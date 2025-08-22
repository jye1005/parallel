#include "Imagelib.h"
#include "CLayer.h"
#include <fstream>
using namespace std;



void print_all_elements(const Tensor3D& tensor) {
	int nH, nW, nC;
	tensor.get_info(nH, nW, nC);
	cout.precision(3);
	for (int c = 0; c < nC; c++) {
		cout << c << "-th channel:" << endl;
		for (int h = 0; h < nH; h++) {
			for (int w = 0; w < nW; w++) {
				cout << setw(8) << tensor.get_elem(h, w, c);
			}
			cout << endl;
		}
	}
}

Tensor3D* read_tensor(const string& filename) {
    ifstream fin(filename);
    if (!fin) {
        cerr << "Failed to open file: " << filename << endl;
        return nullptr;
    }
    int fK = 5;
    int fC = 2;
    Tensor3D* temp = new Tensor3D(fK, fK, fC);

    for (int c = 0; c < fC; c++) {
        for (int w = 0; w < fK; w++) {
            for (int h = 0; h < fK; h++) {
                double val;
                if (!(fin >> val)) {
                    cerr << "Read error at (h=" << h << ", w=" << w << ", c=" << c << ")\n";
                    delete temp;
                    return nullptr;
                }
                temp->set_elem(h, w, c, val);
            }
        }
    }
    return temp;
}

int main() {
	Layer* layer1 = new Layer_Conv("Conv1", 3, 3, 1, MEAN_INIT); 
	Layer* layer2 = new Layer_ReLU("Relu1",1, 1, 1);

	cout << "(Layer information)________________" << endl;
	layer1->print();
	layer2->print();
	cout << endl << endl;
    
	Tensor3D* tensor1 = read_tensor("tensor_5x5x2.txt");
	Tensor3D* tensor2 = layer1->forward(tensor1);
	Tensor3D* tensor3 = layer2->forward(tensor2);
	cout << endl << endl;
	
	cout << "(Tensor information)________________" << endl;
	cout << "[tensor1]:" << endl;
	tensor1->print();
	print_all_elements((*tensor1));
	cout << endl;

	cout << "[tensor2]:" << endl;
	tensor2->print();
	print_all_elements((*tensor2));
	cout << endl;

	cout << "[tensor3]:" << endl;
	tensor3->print();
	print_all_elements((*tensor3));
	cout << endl;

	system("PAUSE");
	return 0;
}
