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
	if( false ) {
		Net net;
		//auto* pfc1 = new FullyConnected_Layer{ 2, 2 };

		net << shared_ptr<Layer>(std::make_shared<FullyConnected_Layer>(2, 2))
				<< shared_ptr<Layer>(std::make_shared<FullyConnected_Layer>(2, 1));
		//net.print_weights();
		vector<float> in11 = { 1, 1 }, out;
		net.forward(&in11[0], &in11[0]+2, out);
		print_vector("out", out);
		for(int k = 0; k != 100; ++k) {
			vector<float> out_grad = { out[0] - 1.0f }, in_grad;
			net.backward(out_grad, in_grad);
			//net.print_weights();
			net.forward(&in11[0], &in11[0]+2, out);
			print_vector("out", out);
		}
	}
	if( false ) {
		Net net;
		net << shared_ptr<Layer>(std::make_shared<FullyConnected_Layer>(2, 2))
				<< shared_ptr<Layer>(std::make_shared<FullyConnected_Layer>(2, 1));
		//net.print_weights();
		float in[][3] = {
			{0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, 1, 1}, 		//	{x1, x2, and(x1, x2)}
		};
		vector<float> out, in_grad;
		for(int k = 0; k != 1000000; ++k) {
			for(int i = 0; i != 4; ++i) {
				net.forward(&in[i][0], &in[i][2], out);
				if( k%100 == 99 )
					print_vector("out", out);
				vector<float> out_grad = { out[0] - in[i][2] };
				net.backward(out_grad, in_grad);
			}
		}
	}
	if( false ) {
		Net net;
		net << shared_ptr<Layer>(std::make_shared<FullyConnected_Layer>(2, 2))
				<< shared_ptr<Layer>(std::make_shared<ReLU_Layer>(2))
				<< shared_ptr<Layer>(std::make_shared<FullyConnected_Layer>(2, 1));
		//net.print_weights();
		float in[][3] = {
			{0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, 1, 1}, 		//	{x1, x2, and(x1, x2)}
		};
		vector<float> out, in_grad;
		for(int k = 0; k != 1000; ++k) {
			if( k%100 == 99 )
				cout << "k = " << k << ":\n";
			for(int i = 0; i != 4; ++i) {
				net.forward(&in[i][0], &in[i][2], out);
				if( k%100 == 99 )
					print_vector("out", out);
				vector<float> out_grad = { out[0] - in[i][2] };
				net.backward(out_grad, in_grad);
			}
		}
	}
	if( false ) {
		Net net;
		net << shared_ptr<Layer>(std::make_shared<FullyConnected_Layer>(2, 2))
				<< shared_ptr<Layer>(std::make_shared<ReLU_Layer>(2))
				<< shared_ptr<Layer>(std::make_shared<FullyConnected_Layer>(2, 1));
		//net.print_weights();
		float in[][3] = {
			{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 1}, 		//	{x1, x2, or(x1, x2)}
		};
		vector<float> out, in_grad;
		for(int k = 0; k != 1000; ++k) {
			if( k%100 == 99 )
				cout << "k = " << k << ":\n";
			for(int i = 0; i != 4; ++i) {
				net.forward(&in[i][0], &in[i][2], out);
				if( k%100 == 99 )
					print_vector("out", out);
				vector<float> out_grad = { out[0] - in[i][2] };
				net.backward(out_grad, in_grad);
			}
		}
	}
	if( false ) {
		Net net;
		net << shared_ptr<Layer>(std::make_shared<FullyConnected_Layer>(2, 2))
				<< shared_ptr<Layer>(std::make_shared<ReLU_Layer>(2))
				<< shared_ptr<Layer>(std::make_shared<FullyConnected_Layer>(2, 1));
		//net.print_weights();
		float in[][3] = {
			{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}, 		//	{x1, x2, xor(x1, x2)}
		};
		vector<float> out, in_grad;
		for(int k = 0; k != 1000; ++k) {
			if( k%100 == 99 )
				cout << "k = " << k << ":\n";
			for(int i = 0; i != 4; ++i) {
				net.forward(&in[i][0], &in[i][2], out);
				if( k%100 == 99 )
					print_vector("out", out);
				vector<float> out_grad = { out[0] - in[i][2] };
				net.backward(out_grad, in_grad);
			}
		}
	}
	if( true ) {
		Net net;
		net << shared_ptr<Layer>(std::make_shared<FullyConnected_Layer>(2, 10))
				<< shared_ptr<Layer>(std::make_shared<ReLU_Layer>(10))
				<< shared_ptr<Layer>(std::make_shared<FullyConnected_Layer>(10, 1));
		//net.print_weights();
		float in[][3] = {
			{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 2}, 		//	{x1, x2, add(x1, x2)}
		};
		vector<float> out, in_grad;
		for(int k = 0; k != 1000; ++k) {
			if( k%100 == 99 )
				cout << "k = " << k << ":\n";
			for(int i = 0; i != 4; ++i) {
				net.forward(&in[i][0], &in[i][2], out);
				if( k%100 == 99 )
					print_vector("out", out);
				vector<float> out_grad = { out[0] - in[i][2] };
				net.backward(out_grad, in_grad);
			}
		}
	}
	//
    std::cout << "\nOK.\n";
}
