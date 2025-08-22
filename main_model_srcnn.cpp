#include "Imagelib.h"
#include "CModel.h"
using namespace std;




int main() {
	Model model;
	
	model.add_layer(new Layer_Conv("Conv1", 9, 1, 64, LOAD_INIT, "model/weights_conv1_9x9x1x64.txt", "model/biases_conv1_64.txt"));
	model.add_layer(new Layer_ReLU("Relu1", 1, 64, 64));
	model.add_layer(new Layer_Conv("Conv2", 5, 64, 32, LOAD_INIT, "model/weights_conv2_5x5x64x32.txt", "model/biases_conv2_32.txt"));
	model.add_layer(new Layer_ReLU("Relu2", 1, 32, 32));
	model.add_layer(new Layer_Conv("Conv3", 5, 32, 1, LOAD_INIT, "model/weights_conv3_5x5x32x1.txt", "model/biases_conv3_1.txt"));


	model.test("baby_512x512_input.bmp", "baby_512x512_output_srcnn.bmp");

	model.print_layer_info();
	model.print_tensor_info();
	system("PAUSE");

	return 0;
}
