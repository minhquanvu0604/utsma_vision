#include <torch/script.h>  // LibTorch
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>

#include "predict.hpp"

class ModelInferenceImpl {
public:
    // Constructor to initialize the model and input size
    ModelInferenceImpl(const std::string& model_path, const cv::Size& input_size) 
        : input_size_(input_size) {
        // Load the model
        try {
            model_ = torch::jit::load(model_path);

            // TODO Check the problem with using CPU for model inference
            // Move the model to GPU if available
            if (torch::cuda::is_available()) {
                model_.to(torch::kCUDA);
                std::cout << "CUDA is available. Moving model to GPU." << std::endl;
            } else {
                model_.to(torch::kCPU);
                std::cout << "CUDA is not available. Moving model to CPU." << std::endl;
            }
            // model_.to(torch::kCPU);


        } catch (const c10::Error& e) {
            std::cerr << "Error loading the model from " << model_path << std::endl;
            throw;
        }

        std::cout << "Model loaded successfully from " << model_path << std::endl;
    }

    // Function to infer a single image
    cv::Mat infer_single_image(const cv::Mat& image) {

        // std::cout << "[DEBUG MODE] INFERING A HARD CODED IMAGE" << std::endl;
        // cv::Mat image = cv::imread("/home/quanvu/Desktop/apple_gazebo/apples_gazebo.png");

        if (image.empty()) {
            std::cerr << "Error: Provided image is empty!" << std::endl;
            return cv::Mat();
        }

        // Preprocess the image
        auto image_tensor = preprocess_image(image, input_size_);

        if (!image_tensor.defined()) {
            std::cerr << "Error during image preprocessing!" << std::endl;
            return cv::Mat();
        }

        // Inference
        auto probability_map = infer(model_, image_tensor, image.size());

        // Return the probability map
        return probability_map;
    }

    // Preprocess the image
    torch::Tensor preprocess_image(cv::Mat image, const cv::Size& input_size) {
        // Convert the image to float32 and scale values to [0, 1]
        image.convertTo(image, CV_32F, 1.0 / 255.0);
        cv::resize(image, image, input_size);
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

        auto tensor_image = torch::from_blob(image.data, {1, image.rows, image.cols, 3}, torch::kFloat32);
        tensor_image = tensor_image.permute({0, 3, 1, 2});  // Change to CxHxW

        // Normalize the image using the mean and std from torchvision
        tensor_image = tensor_image.sub_(torch::tensor({0.485, 0.456, 0.406}).view({1, 3, 1, 1}))
                                    .div_(torch::tensor({0.229, 0.224, 0.225}).view({1, 3, 1, 1}));

        return tensor_image.clone();  // Return a deep copy of the tensor
    }

    // Run inference and get probability map
    cv::Mat infer(torch::jit::script::Module& model, torch::Tensor& image_tensor, const cv::Size& original_size) {
        model.eval();  // Ensure the model is in evaluation mode

        // Move the image tensor to GPU if available
        if (torch::cuda::is_available())
            image_tensor = image_tensor.to(torch::kCUDA);
        else
            image_tensor = image_tensor.to(torch::kCPU);
        // image_tensor = image_tensor.to(torch::kCPU);

        // Inference
        torch::NoGradGuard no_grad;  // Disable gradient computation
        auto output = model.forward({image_tensor});  // Perform forward pass
        torch::cuda::synchronize();  // After the forward pass

        // Access the "out" key from the output dictionary (DeepLabV3 specific)
        auto out_tensor = output.toGenericDict().at("out").toTensor();

        // Get softmax probabilities and extract the apple class (index 1)
        auto probabilities = torch::softmax(out_tensor, 1);
        auto apple_prob = probabilities[0][1].unsqueeze(0);  // Take the class 1 (apples) and keep as 3D

        // Fix: Add an extra dimension to make it a 4D tensor
        apple_prob = apple_prob.unsqueeze(0); // Shape becomes [1, 1, height, width]

        // Resize the probabilities to original size
        auto resized_prob = torch::nn::functional::interpolate(
            apple_prob,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{original_size.height, original_size.width})
                .mode(torch::kBilinear)
                .align_corners(false)
        );

        // Convert to CPU and extract the data as a cv::Mat
        resized_prob = resized_prob.squeeze().cpu();  // Remove batch and channel dimensions

        // Create cv::Mat with float data from the tensor
        // It’s possible that cv::Mat is behaving unexpectedly in your case because the OpenCV matrix 
        // probability_map directly wraps the underlying Torch tensor data with shared memory
        // When using cv::Mat as a wrapper around a Torch tensor, any changes made to the Torch tensor can 
        // reflect in cv::Mat because they both reference the same memory. 
        // This means that if infer() modifies or reuses the tensor resized_prob after wrapping it in cv::Mat, 
        // the values in probability_map will also change.
        cv::Mat probability_map(original_size, CV_32F, resized_prob.data_ptr<float>());

        // return probability_map; // Return the cv::Mat directly - above problem
        return probability_map.clone();  // Return a deep copy of the cv::Mat
    }

private:
    torch::jit::script::Module model_;
    cv::Size input_size_;
};

ModelInference::ModelInference(const std::string& model_path, const cv::Size& input_size)
    : impl_(std::make_unique<ModelInferenceImpl>(model_path, input_size)) {}

ModelInference::~ModelInference() = default;

cv::Mat ModelInference::infer_single_image(const cv::Mat& image) {
    return impl_->infer_single_image(image);
}
