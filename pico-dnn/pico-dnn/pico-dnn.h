//----------------------------------------------------------------------
//
//			File:			"pico-dnn.h"
//			Created:		17-3-2023
//			Author:			津田伸秀
//			Description:
//
//----------------------------------------------------------------------

#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <memory>

//	レイヤー（層）基底クラス
class Layer {
public:
    Layer() {}
    virtual ~Layer() {}
    // 順伝播（forward propagation）
    virtual void forward(const float* input, std::vector<float>& output) = 0;
    //virtual void forward(const std::vector<float>& input, std::vector<float>& output) = 0;
	// 逆伝播（backward propagation）
    virtual void backward(const std::vector<float>& output_grad, std::vector<float>& input_grad, bool online=true) = 0;
    //	ミニバッチ重み更新
    virtual void update(int BSZ) = 0;
    // 重み表示
    virtual void print_weights() const = 0;
};

//	総結合層
class FullyConnected_Layer : public Layer {
public:
    FullyConnected_Layer(int input_size, int output_size) :
        input_size_(input_size),
        output_size_(output_size),
        weights_(output_size, std::vector<float>(input_size, 0)),
        dweights_(output_size, std::vector<float>(input_size, 0)),
        bias_(output_size, 0),
        dbias_(output_size, 0),
        input_(input_size, 0),
        output_(output_size, 0),
        grad_weights_(output_size, std::vector<float>(input_size, 0)),
        grad_bias_(output_size, 0),
        output_grad_(output_size, 0),
        input_grad_(input_size, 0),
        rng_(std::random_device{}())
    {
        // 重みとバイアスをランダムに初期化する
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < input_size; ++j) {
                weights_[i][j] = dist(rng_);
            }
            bias_[i] = dist(rng_);
        }
    }

    virtual void forward(const float* input, std::vector<float>& output) override {
        input_ = std::vector<float>(input, input + input_size_);
        for (int i = 0; i < output_size_; ++i) {
            float dot = 0;
            for (int j = 0; j < input_size_; ++j) {
                dot += weights_[i][j] * input[j];
            }
            output_[i] = dot + bias_[i];
        }
        output = output_;
    }
#if 0
    virtual void forward(const std::vector<float>& input, std::vector<float>& output) override {
        input_ = input;
        for (int i = 0; i < output_size_; ++i) {
            float dot = 0;
            for (int j = 0; j < input_size_; ++j) {
                dot += weights_[i][j] * input[j];
            }
            output_[i] = dot + bias_[i];
        }
        output = output_;
    }
#endif

    virtual void backward(const std::vector<float>& output_grad, std::vector<float>& input_grad, bool online) override {
        // バイアスの勾配を計算する
        for (int i = 0; i < output_size_; ++i) {
            grad_bias_[i] = output_grad[i];
        }
        // 重みと入力の勾配を計算する
        for (int j = 0; j < input_size_; ++j) {
            float dot = 0;
            for (int i = 0; i < output_size_; ++i) {
                grad_weights_[i][j] = output_grad[i] * input_[j];
                dot += output_grad[i] * weights_[i][j];
            }
            input_grad_[j] = dot;
        }
        // 勾配を更新する
        if( online ) {
	        for (int i = 0; i < output_size_; ++i) {
	            for (int j = 0; j < input_size_; ++j) {
	                weights_[i][j] -= learning_rate_ * grad_weights_[i][j];
	            }
	            bias_[i] -= learning_rate_ * grad_bias_[i];
	        }
        } else {
	        for (int i = 0; i < output_size_; ++i) {
	            for (int j = 0; j < input_size_; ++j) {
	                dweights_[i][j] -= learning_rate_ * grad_weights_[i][j];
	            }
	            dbias_[i] -= learning_rate_ * grad_bias_[i];
	        }
        }
        input_grad = input_grad_;
    }
    virtual void update(int BSZ) {
        for (int i = 0; i < output_size_; ++i) {
            for (int j = 0; j < input_size_; ++j) {
                weights_[i][j] -= dweights_[i][j] / BSZ;
                dweights_[i][j] = 0;
            }
            bias_[i] -= dbias_[i] / BSZ;
            dbias_[i] = 0;
        }
    }
    virtual void print_weights() const {
        for (int i = 0; i < output_size_; ++i) {
            std::cout << "{";
            for (int j = 0; j < input_size_; ++j) {
                std::cout << weights_[i][j] << ", ";
            }
            std::cout << "}\n";
        }
    }
private:
    int input_size_; // 入力の次元数
    int output_size_; // 出力の次元数
    std::vector<std::vector<float>> weights_; // 重み
    std::vector<std::vector<float>> dweights_; // 重み
    std::vector<float> bias_; // バイアス
    std::vector<float> dbias_; // バイアス差分
    std::vector<float> input_; // 入力
    std::vector<float> output_; // 出力
    std::vector<std::vector<float>> grad_weights_; // 重みの勾配
	std::vector<float> grad_bias_; // バイアスの勾配
    std::vector<float> output_grad_; // 出力の勾配
    std::vector<float> input_grad_; // 入力の勾配
    std::mt19937 rng_; // 乱数生成器
    //const float learning_rate_ = 0.01f; // 学習率
    const float learning_rate_ = 0.1f; // 学習率
};

class ReLU_Layer : public Layer {
public:
    ReLU_Layer(int input_size) : input_size_(input_size) {}
    virtual ~ReLU_Layer() {}

    virtual void forward(const float* input, std::vector<float>& output) override {
        output.resize(input_size_);
        output_.resize(input_size_);
        for (int i = 0; i < input_size_; i++) {
            output[i] = output_[i] = std::max(0.0f, input[i]);
        }
    }
#if 0
    void forward(const std::vector<float>& input, std::vector<float>& output) override {
        // ReLU関数を適用する
        output.resize(input.size());
        output_.resize(input.size());
        for (int i = 0; i < input.size(); i++) {
            output[i] = output_[i] = std::max(0.0f, input[i]);
        }
    }
#endif

    void backward(const std::vector<float>& output_grad, std::vector<float>& input_grad, bool) override {
        // ReLU関数の逆伝播を計算する
        input_grad.resize(output_grad.size());
        for (int i = 0; i < output_grad.size(); i++) {
            input_grad[i] = output_grad[i] * (output_[i] > 0);
        }
    }
    virtual void update(int BSZ) {}
    virtual void print_weights() const {}
private:
    int	input_size_;
	std::vector<float> output_;
};
class SoftMax_Layer : public Layer {
public:
    SoftMax_Layer(int input_size) : input_size_(input_size) {}
    virtual ~SoftMax_Layer() {}

    void forward(const float* input, std::vector<float>& output) override {
        // Softmax関数を適用する
        output.resize(input_size_);
        output_.resize(input_size_);
        auto mx = *std::max_element(input, input + input_size_);
        float sum = 0.0f;
        for (int i = 0; i < input_size_; i++) {
            output_[i] = std::exp(input[i] - mx);
            sum += output_[i];
        }
        for (int i = 0; i < output.size(); i++) {
            output[i] = output_[i] /= sum;
        }
    }
    void backward(const std::vector<float>& output_grad, std::vector<float>& input_grad, bool) override {
        // Softmax関数の逆伝播を計算する。ただし grad をスルーするだけ
#if 1
        input_grad = output_grad;
#else
        input_grad.resize(output_grad.size());
        float sum = 0.0f;
        for (int i = 0; i < output_grad.size(); i++) {
            sum += output_[i] * output_grad[i];
        }
        for (int i = 0; i < input_grad.size(); i++) {
            input_grad[i] = output_[i] * (output_grad[i] - sum);
        }
#endif
    }
    virtual void update(int BSZ) {}
    virtual void print_weights() const {}
private:
    int	input_size_;
	std::vector<float> output_;
};
#if 1
//	正規化レイヤー
class Normalization_Layer : public Layer {
public:
    Normalization_Layer(int input_size, float eps = 1e-5) : m_input_size(input_size), m_eps(eps) {}
    virtual ~Normalization_Layer() {}
    
    virtual void forward(const float* input, std::vector<float>& output) override {
        // バッチ平均とバッチ分散を計算する
        m_input.resize(m_input_size);
        output.resize(m_input_size);
        m_mean = 0.0;		//	平均
        m_var = 0.0;		//	分散
        for (int i = 0; i < m_input_size; i++)
            m_mean += m_input[i] = *input++;
        m_mean /= m_input_size;
        for (int i = 0; i < m_input_size; i++)
            m_var += (input[i] - m_mean) * (input[i] - m_mean);
        for (int i = 0; i < m_input_size; i++)
	        output[i] = (input[i] - m_mean) / (std::sqrt(m_var + m_eps));
    }

    virtual void backward(const std::vector<float>& output_grad, std::vector<float>& input_grad, bool online=true) override {
        // 入力勾配を計算する
        input_grad.resize(m_input_size);
        for (int i = 0; i < m_input_size; i++) {
            float z = (m_input[i] - m_mean) / std::sqrt(m_var + m_eps);
            input_grad[i] = output_grad[i] / std::sqrt(m_var + m_eps) - z * (output_grad[i] * z) / std::sqrt(m_var + m_eps);
        }
    }
    
    virtual void update(int BSZ) override {}
    virtual void print_weights() const {}

protected:
    std::vector<float>	m_input;
    float m_eps;
    int m_input_size;
    float m_mean;		//	平均
    float m_var;			//	分散
};
#endif
#if 0
//	正則化レイヤー
class RegularizationLayer : public Layer {
public:
    RegularizationLayer(float l1_coef, float l2_coef) : l1_coef_(l1_coef), l2_coef_(l2_coef) {}
    virtual ~RegularizationLayer() {}
    
    virtual void forward(const float* input, std::vector<float>& output) override {
        // 正則化項を計算し、出力に加える
        float l1_reg = 0.0;
        float l2_reg = 0.0;
        for (int i = 0; i < input_size_; i++) {
            l1_reg += std::abs(input[i]);
            l2_reg += input[i] * input[i];
        }
        output.resize(input_size_);
        for (int i = 0; i < input_size_; i++) {
            output[i] = input[i] + l1_coef_ * l1_reg + l2_coef_ * l2_reg;
        }
    }

    virtual void backward(const std::vector<float>& output_grad, std::vector<float>& input_grad, bool online=true) override {
        // 入力勾配を計算する
        input_grad.resize(input_size_);
        for (int i = 0; i < input_size_; i++) {
            float sign = input[i] > 0 ? 1.0 : -1.0;
            input_grad[i] = output_grad[i] + l1_coef_ * sign + 2 * l2_coef_ * input[i];
        }
    }
    
    virtual void update(int BSZ) override {}

protected:
    float l1_coef_;
    float l2_coef_;
    int input_size_;
};
#endif
class ConvolutionalLayer : public Layer {
public:
    ConvolutionalLayer(int input_height, int input_width, /*int input_channels,*/ int kernel_size, int num_kernels) :
        input_height_(input_height),
        input_width_(input_width),
        //input_channels_(input_channels),		//	チャンネル数は１のみとする
        kernel_size_(kernel_size),			//	カーネル（フィルター）サイズ、正方形の一辺長さ
        num_kernels_(num_kernels)
    {
        // 畳み込みカーネルをランダムに初期化
        weights_.resize(num_kernels_);
        for (int i = 0; i < num_kernels_; ++i) {
            weights_[i].resize(kernel_size_ * kernel_size_ /** input_channels_*/);
            for (int j = 0; j < kernel_size_ * kernel_size_ /** input_channels_*/; ++j) {
                //##weights_[i][j] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
            }
        }

        // バイアスを0で初期化
        biases_.resize(num_kernels_, 0.0f);

        // 出力テンソルのサイズを計算
        output_height_ = input_height_ - kernel_size_ + 1;
        output_width_ = input_width_ - kernel_size_ + 1;
        output_channels_ = num_kernels_;
    }

    virtual ~ConvolutionalLayer() {}

    virtual void forward(const float* input, std::vector<float>& output) override {
#if 0
        output.resize(output_height_ * output_width_ * output_channels_);
        // 入力テンソルを3次元配列に変換
        std::vector<std::vector<std::vector<float>>> input_array(input_channels_,
            std::vector<std::vector<float>>(input_height_,
                std::vector<float>(input_width_, 0.0f)));
        int input_index = 0;
        for (int c = 0; c < input_channels_; ++c) {
            for (int h = 0; h < input_height_; ++h) {
                for (int w = 0; w < input_width_; ++w) {
                    input_array[c][h][w] = input[input_index++];
                }
            }
        }

        // 畳み込み演算を実行
        for (int k = 0; k < num_kernels_; ++k) {
            for (int h = 0; h < output_height_; ++h) {
                for (int w = 0; w < output_width_; ++w) {
                    float sum = 0.0f;
                    for (int c = 0; c < input_channels_; ++c) {
                        for (int kh = 0; kh < kernel_size_; ++kh) {
                            for (int kw = 0; kw < kernel_size_; ++kw) {
                                int input_h = h + kh;
                                int input_w = w + kw;
                                sum += input_array[c][input_h][input_w] * weights_[k][c * kernel_size_ * kernel_size_ + kh * kernel_size_ + kw];
                            }
                        }
                    }
                    output[k * output_height_ * output_width_ + h * output_width_ + w] = sum + biases_[k];
                }
            }
        }
#endif
    }

    virtual void backward(const std::vector<float>& output_grad, std::vector<float>& input_grad, bool online = true) override {
        //input_grad.resize(input_height_ * input_width_
    }
private:
    int input_width_;		//	入力画像幅
    int input_height_;		//	入力画像高さ
    int input_channels_;	// 入力チャンネル数
    int out_channels_;		// 出力チャンネル数
    int kernel_size_;		// 畳み込みカーネルのサイズ（正方形）
    int num_kernels_;
    int stride_;			// ストライドのサイズ
    int padding_;			// パディングのサイズ
    int output_width_;
    int output_height_;
    int output_channels_;
    std::vector<std::vector<std::vector<float>>> weights_; // 畳み込みカーネルの重み
    std::vector<float> biases_; // バイアス
};
//----------------------------------------------------------------------
class Net {
public:
    Net() {}
    virtual ~Net() {}

    // ネットワークにレイヤーを追加する
    void add_layer(std::shared_ptr<Layer> layer) {
        layers_.push_back(layer);
    }
	Net& operator<<(std::shared_ptr<Layer> layer) {
        add_layer(layer);
        return *this;
    }
    // 順伝播（forward propagation）
    void forward(const float* input, const float* input_end, std::vector<float>& output) {
        //const float* temp_input = input;
        std::vector<float> temp_input = std::vector<float>(input, input_end);
        for (auto layer : layers_) {
            layer->forward(&temp_input[0], output);
            temp_input = output;
        }
    }
#if 0
    void forward(const std::vector<float>& input, std::vector<float>& output) {
        std::vector<float> temp_input = input;
        for (auto layer : layers_) {
            layer->forward(temp_input, output);
            temp_input = output;
        }
    }
#endif

    // 逆伝播（backward propagation）
    void backward(const std::vector<float>& output_grad, std::vector<float>& input_grad, bool online=true) {
        std::vector<float> temp_output_grad = output_grad;
        for (int i = (int)layers_.size() - 1; i >= 0; --i) {
            std::vector<float> temp_input_grad;
            layers_[i]->backward(temp_output_grad, temp_input_grad, online);
            temp_output_grad = temp_input_grad;
        }
        input_grad = temp_output_grad;
    }
    void update(int BSZ) {
        for (int i = (int)layers_.size() - 1; i >= 0; --i) {
            layers_[i]->update(BSZ);
        }
    }
    void print_weights() const {
	    for (auto layer : layers_) {
	    	layer->print_weights();
	    }
    }

private:
    std::vector<std::shared_ptr<Layer>> layers_;
};
