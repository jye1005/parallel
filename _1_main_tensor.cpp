#include "Imagelib.h"
#include "CTensor.h"
using namespace std;


void print_all_elements(const Tensor3D& tensor) {
	int nH, nW, nC;
	tensor.get_info(nH, nW, nC); 
	for (int c = 0; c < nC; c++) {
		cout << c << "-th channel:" << endl;
		for (int h = 0; h < nH; h++) {
			for (int w = 0; w < nW; w++) {
				cout << setw(4) << tensor.get_elem(h, w, c);
			}
			cout << endl;
		}
	}
}

int main() {

	int nH = 2;
	int nW = 3;
	int nC = 2;
	Tensor3D tensor(nH, nW, nC); 

	
	tensor.set_elem(0, 0, 0, 1);
	tensor.set_elem(1, 1, 0, 2);
	tensor.set_elem(0, 0, 1, 3);
	tensor.set_elem(1, 2, 1, 4);

	print_all_elements(tensor);

	tensor.print();

	system("PAUSE");

	return 0;
}
