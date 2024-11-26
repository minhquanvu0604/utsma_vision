#ifndef PREDICT_H
#define PREDICT_H

#include <opencv2/opencv.hpp>
#include <memory>
#include <string>


class ModelInferenceImpl; // Forward declaration of the implementation class

class ModelInference {
public:
    // Constructor to initialize the model and input size
    ModelInference(const std::string& model_path, const cv::Size& input_size);

    // Destructor
    ~ModelInference();

    // Function to infer a single image
    cv::Mat infer_single_image(const cv::Mat& image);

private:
    std::unique_ptr<ModelInferenceImpl> impl_; // Pointer to the implementation
};

#endif // PREDICT_H
