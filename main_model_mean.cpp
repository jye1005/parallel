#include "Imagelib.h"
#include "CModel.h"
using namespace std;


int main() {
	Model model;

	// build model
	model.add_layer(new Layer_Conv("Conv1", 3, 1, 1, MEAN_INIT));
	model.add_layer(new Layer_ReLU("Relu1", 1, 1, 1));
	model.add_layer(new Layer_Conv("Conv2", 3, 1, 1, MEAN_INIT));
	model.add_layer(new Layer_ReLU("Relu2", 1, 1, 1));
	model.add_layer(new Layer_Conv("Conv3", 3, 1, 1, MEAN_INIT));


	model.test("baby_512x512_input.bmp", "baby_512x512_output_mean.bmp");

	model.print_layer_info();
	model.print_tensor_info();
	system("PAUSE");

	return 0;
}
