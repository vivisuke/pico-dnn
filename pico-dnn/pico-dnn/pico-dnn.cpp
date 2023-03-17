#include <iostream>
#include "pico-dnn.h"

using namespace std;

void print_vector(const string& name, const vector<float>& v) {
	cout << name << "[] = {";
	for (int i = 0; i != v.size(); ++i) cout << v[i] << ", ";
	cout << "}\n";
}
int main()
{
	if( false ) {
		FullyConnected_Layer fc1(2, 2);
		FullyConnected_Layer fc2(2, 1);
	}
	if( true ) {
		Net net;
		//auto* pfc1 = new FullyConnected_Layer{ 2, 2 };

		net << shared_ptr<Layer>(std::make_shared<FullyConnected_Layer>(2, 2))
				<< shared_ptr<Layer>(std::make_shared<FullyConnected_Layer>(2, 1));
		//net.print_weights();
		vector<float> in11 = { 1, 1 }, out;
		net.forward(in11, out);
		print_vector("out", out);
		for(int k = 0; k != 100; ++k) {
			vector<float> out_grad = { out[0] - 1.0f }, in_grad;
			net.backward(out_grad, in_grad);
			//net.print_weights();
			net.forward(in11, out);
			print_vector("out", out);
		}
	}
	//
    std::cout << "\nOK.\n";
}
