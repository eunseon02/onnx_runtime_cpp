/* ASpanFormer.hpp */
#pragma once

#include <ort_utility/ort_utility.hpp>

namespace Ort {
class ASpanFormer : public OrtSessionHandler 
{
public:
    // Input image dimensions (must match exported ONNX model)
    static constexpr int64_t IMG_H = 480;
    static constexpr int64_t IMG_W = 640;
    static constexpr int64_t IMG_CHANNEL = 1;

    using OrtSessionHandler::OrtSessionHandler;

    // Normalize unsigned char image to float [0,1]
    void preprocess(float* dst, const unsigned char* src,
                    const int64_t targetImgWidth, const int64_t targetImgHeight,
                    const int numChannels) const;
};
}  // namespace Ort


