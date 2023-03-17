//----------------------------------------------------------------------
//
//			File:			"pico-dnn.h"
//			Created:		17-3-2023
//			Author:			�Óc�L�G
//			Description:
//
//----------------------------------------------------------------------

#pragma once

#include <vector>
#include <random>

//	���C���[�i�w�j���N���X
class Layer {
public:
    Layer() {}
    virtual ~Layer() {}
    // ���`�d�iforward propagation�j
    virtual void forward(const std::vector<float>& input, std::vector<float>& output) = 0;
	// �t�`�d�ibackward propagation�j
    virtual void backward(const std::vector<float>& output_grad, std::vector<float>& input_grad) = 0;
};

