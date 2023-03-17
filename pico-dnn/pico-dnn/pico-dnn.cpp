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
		vector<float> in = { 1, 1 }, out;
		net.forward(in, out);
		print_vector("out", out);
		print_vector("out", out);
		//cout << "out[] = {";
		//for (int i = 0; i != out.size(); ++i) cout << out[i] << ", ";
		//cout << "}\n";
	}
	//
    std::cout << "\nOK.\n";
}
