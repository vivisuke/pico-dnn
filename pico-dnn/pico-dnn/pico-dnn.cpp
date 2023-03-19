#include <iostream>
#include <fstream>      // std::ifstream
#include <string>
#include <chrono>
#include "pico-dnn.h"

using namespace std;

const int MNIST_IMG_WD = 28;
const int MNIST_IMG_HT = 28;
const int MNIST_IMG_SZ = MNIST_IMG_WD*MNIST_IMG_HT;
const int IMG_HDR_SIZE = 16;
const int LBL_HDR_SIZE = 8;

typedef unsigned char uchar;

//	最大値を与える要素インデックスを返す
//	ただし、v[i] は [0, 1] の範囲とする
int max_index(const vector<float>& v) {
	float mx = -1.0f;
	int mi = -1;
	for (int i = 0; i != v.size(); ++i) {
		if( v[i] > mx ) {
			mx = v[i];
			mi = i;
		}
	}
	return mi;		
}
void print_vector(const string& name, const vector<float>& v) {
	cout << name << "[] = {";
	float mx = -1.0f;
	int mi = 0;
	for (int i = 0; i != v.size(); ++i) {
		if( v[i] > mx ) {
			mx = v[i];
			mi = i;
		}
		string txt = to_string(v[i]);
		if( txt.size() > 5 ) txt = txt.substr(0, 5);
		cout << txt << ", ";
		//cout << v[i] << ", ";
	}
	cout << "}, max ix = " << mi << "\n";
	//cout << "}\n";
}
void print_image(const uchar* ptr) {
	for(int y = 0; y != MNIST_IMG_HT; ++y) {
		for(int x = 0; x != MNIST_IMG_WD; ++x) {
			auto ch = *ptr++;
			if( ch == 0 ) cout << "・";
			else cout << "■";
		}
		cout << "\n";
	}
}
int main()
{
	auto start = std::chrono::system_clock::now();      // 計測スタート時刻を保存
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
	if( false ) {
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
	if( false ) {
		Net net;
		net << shared_ptr<Layer>(std::make_shared<FullyConnected_Layer>(4, 10))
				<< shared_ptr<Layer>(std::make_shared<ReLU_Layer>(10))
				<< shared_ptr<Layer>(std::make_shared<FullyConnected_Layer>(10, 3))
				<< shared_ptr<Layer>(std::make_shared<SoftMax_Layer>(3));
		float in[][5] = {
			{0, 0, 1, 0, 0, }, 		//	{0|1}..., {0|1|2}
		};
		cout << "t = " << (int)in[0][4] << "\n";
		vector<float> out, in_grad;
		for(int i = 0; i != 20; ++i) {
			net.forward(&in[0][0], &in[0][5], out);
			print_vector("out", out);
			vector<float> out_grad = out;
			out_grad[(int)in[0][4]] -= 1.0f;
			net.backward(out_grad, in_grad);
		}
	}
	if( false ) {
		Net net;
		net << shared_ptr<Layer>(std::make_shared<FullyConnected_Layer>(MNIST_IMG_SZ, 1024))
				<< shared_ptr<Layer>(std::make_shared<ReLU_Layer>(1024))
				//<< shared_ptr<Layer>(std::make_shared<FullyConnected_Layer>(1024, 1024))
				//<< shared_ptr<Layer>(std::make_shared<ReLU_Layer>(1024))
				<< shared_ptr<Layer>(std::make_shared<FullyConnected_Layer>(1024, 10))
				<< shared_ptr<Layer>(std::make_shared<SoftMax_Layer>(10));
	    string images_path = "g:/data_set/MNIST/train-images.idx3-ubyte";
	    std::ifstream ifs_images(images_path, std::ios::binary);
	    std::istreambuf_iterator<char> it_ifsi_begin(ifs_images);
	    std::istreambuf_iterator<char> it_ifsi_end{};
	    std::vector<unsigned char> images_data(it_ifsi_begin, it_ifsi_end);
	    cout << "images data size = " << images_data.size() << "\n";
	    string labels_path = "g:/data_set/MNIST/train-labels.idx1-ubyte";
	    std::ifstream ifs_labels(labels_path, std::ios::binary);
	    std::istreambuf_iterator<char> it_ifsl_begin(ifs_labels);
	    std::istreambuf_iterator<char> it_ifsl_end{};
	    std::vector<unsigned char> labels_data(it_ifsl_begin, it_ifsl_end);
	    cout << "labels data size = " << labels_data.size() << "\n";
		//	先頭10個のデータで100回学習
		vector<float> in(MNIST_IMG_SZ), out, in_grad;
		for(int k = 0; k != 100; ++k) {
		    for(int i = 0; i != 10; ++i) {
		    	uchar *ptr = &images_data[i*MNIST_IMG_SZ + IMG_HDR_SIZE];
		    	//print_image(ptr);
		    	for(int j = 0; j != MNIST_IMG_SZ; ++j) in[j] = (float)*ptr++ / 0x100;
				net.forward(&in[0], &in[0] + MNIST_IMG_SZ, out);
				//if( k % 10 == 9 )
				if( true )
				{
					if( i == 0 )
				    	cout << "k = " << (k+1) << "\n";
			    	cout << "label = " << (int)labels_data[i + LBL_HDR_SIZE] << "\n";
					print_vector("out", out);
				}
				vector<float> out_grad = out;
				out_grad[(int)labels_data[i + LBL_HDR_SIZE]] -= 1.0f;
				net.backward(out_grad, in_grad);
		    }
		}
	}
	if( false ) {
		Net net;
		net << shared_ptr<Layer>(std::make_shared<FullyConnected_Layer>(MNIST_IMG_SZ, 1024))
				<< shared_ptr<Layer>(std::make_shared<ReLU_Layer>(1024))
				//<< shared_ptr<Layer>(std::make_shared<FullyConnected_Layer>(1024, 1024))
				//<< shared_ptr<Layer>(std::make_shared<ReLU_Layer>(1024))
				<< shared_ptr<Layer>(std::make_shared<FullyConnected_Layer>(1024, 10))
				<< shared_ptr<Layer>(std::make_shared<SoftMax_Layer>(10));
	    string images_path = "g:/data_set/MNIST/train-images.idx3-ubyte";
	    std::ifstream ifs_images(images_path, std::ios::binary);
	    std::istreambuf_iterator<char> it_ifsi_begin(ifs_images);
	    std::istreambuf_iterator<char> it_ifsi_end{};
	    std::vector<unsigned char> images_data(it_ifsi_begin, it_ifsi_end);
	    cout << "images data size = " << images_data.size() << "\n";
	    string labels_path = "g:/data_set/MNIST/train-labels.idx1-ubyte";
	    std::ifstream ifs_labels(labels_path, std::ios::binary);
	    std::istreambuf_iterator<char> it_ifsl_begin(ifs_labels);
	    std::istreambuf_iterator<char> it_ifsl_end{};
	    std::vector<unsigned char> labels_data(it_ifsl_begin, it_ifsl_end);
	    cout << "labels data size = " << labels_data.size() << "\n";

		//	全データ（６万）で1回学習
		int n_correct = 0;
		vector<float> in(MNIST_IMG_SZ), out, in_grad;
	    for(int i = 0; i != 60000; ++i) {
	    	int t = (int)labels_data[i + LBL_HDR_SIZE];	//	教師値
	    	uchar *ptr = &images_data[i*MNIST_IMG_SZ + IMG_HDR_SIZE];	//	当該画像データ
	    	//print_image(ptr);
	    	for(int j = 0; j != MNIST_IMG_SZ; ++j) in[j] = (float)*ptr++ / 0x100;
			net.forward(&in[0], &in[0] + MNIST_IMG_SZ, out);
			int mi = max_index(out);
			if( mi == t ) n_correct += 1;
			if( i % 1000 == 999 )
			{
		    	cout << "image: #" << (i+1) << "\n";
		    	cout << "label = " << t << "\n";
				print_vector("out", out);
				cout << "correct rate = " << n_correct/10.0 << "%\n";
				n_correct = 0;
			}
			vector<float> out_grad = out;
			out_grad[(int)labels_data[i + LBL_HDR_SIZE]] -= 1.0f;
			net.backward(out_grad, in_grad);
	    }
	}
	if( true ) {
		Net net;
		net << shared_ptr<Layer>(std::make_shared<FullyConnected_Layer>(MNIST_IMG_SZ, 1024))
				<< shared_ptr<Layer>(std::make_shared<ReLU_Layer>(1024))
				//<< shared_ptr<Layer>(std::make_shared<FullyConnected_Layer>(1024, 1024))
				//<< shared_ptr<Layer>(std::make_shared<ReLU_Layer>(1024))
				<< shared_ptr<Layer>(std::make_shared<FullyConnected_Layer>(1024, 10))
				<< shared_ptr<Layer>(std::make_shared<SoftMax_Layer>(10));
	    string images_path = "g:/data_set/MNIST/train-images.idx3-ubyte";
	    std::ifstream ifs_images(images_path, std::ios::binary);
	    std::istreambuf_iterator<char> it_ifsi_begin(ifs_images);
	    std::istreambuf_iterator<char> it_ifsi_end{};
	    std::vector<unsigned char> images_data(it_ifsi_begin, it_ifsi_end);
	    cout << "images data size = " << images_data.size() << "\n";
	    string labels_path = "g:/data_set/MNIST/train-labels.idx1-ubyte";
	    std::ifstream ifs_labels(labels_path, std::ios::binary);
	    std::istreambuf_iterator<char> it_ifsl_begin(ifs_labels);
	    std::istreambuf_iterator<char> it_ifsl_end{};
	    std::vector<unsigned char> labels_data(it_ifsl_begin, it_ifsl_end);
	    cout << "labels data size = " << labels_data.size() << "\n";

		//	全データ（６万）で1回学習（１エポック）、ミニバッチ（バッチサイズ：100）
		const int BATCH_SIZE = 100;
		int n_correct = 0;
		vector<float> in(MNIST_IMG_SZ), out, in_grad;
	    for(int i = 0; i != 60000; ++i) {
	    	int t = (int)labels_data[i + LBL_HDR_SIZE];	//	教師値
	    	uchar *ptr = &images_data[i*MNIST_IMG_SZ + IMG_HDR_SIZE];	//	当該画像データ
	    	//print_image(ptr);
	    	for(int j = 0; j != MNIST_IMG_SZ; ++j) in[j] = (float)*ptr++ / 0x100;
			net.forward(&in[0], &in[0] + MNIST_IMG_SZ, out);
			int mi = max_index(out);
			if( mi == t ) n_correct += 1;
			if( i % 1000 == 999 )
			{
		    	cout << "image: #" << (i+1) << "\n";
		    	cout << "label = " << t << "\n";
				print_vector("out", out);
				cout << "correct rate = " << n_correct/10.0 << "%\n";
				n_correct = 0;
			}
			vector<float> out_grad = out;
			out_grad[(int)labels_data[i + LBL_HDR_SIZE]] -= 1.0f;
			net.backward(out_grad, in_grad);
			if( i % BATCH_SIZE == BATCH_SIZE - 1 )
				net.update(BATCH_SIZE);
	    }
	}
	auto end = std::chrono::system_clock::now();       // 計測終了時刻を保存
    auto dur = end - start;        // 要した時間を計算
    auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
	//
    std::cout << "\nOK.\n";
    cout << "duration = " << msec << "millisec.\n";
}














