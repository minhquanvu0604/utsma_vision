#include <iostream>
#include <filesystem> // C++17 for directory traversal
#include <memory>

#include <torch/script.h> // LibTorch
#include <torch/torch.h>

#include <opencv2/opencv.hpp> // For image loading and preprocessing
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"


torch::Tensor preprocess_image(cv::Mat& image, const cv::Size& input_size) {
    // cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);

    // Convert the image to float32 and scale values to [0, 1]
    image.convertTo(image, CV_32F, 1.0 / 255.0);
    cv::resize(image, image, input_size);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    auto tensor_image = torch::from_blob(image.data, {1, image.rows, image.cols, 3}, torch::kFloat32);

    // Normalize the image using the mean and std from torchvision
    tensor_image = tensor_image.permute({0, 3, 1, 2}); // Change to CxHxW

    // Unsqueeze the normalization tensors to make them compatible with the 4D image tensor
    tensor_image = tensor_image.sub_(torch::tensor({0.485, 0.456, 0.406}).view({1, 3, 1, 1}))
                             .div_(torch::tensor({0.229, 0.224, 0.225}).view({1, 3, 1, 1}));

    // Debugging: Print out the tensor's statistics after normalization
    // std::cout << "Image tensor dtype: " << tensor_image.dtype() << std::endl;
    // std::cout << "Image tensor shape: " << tensor_image.sizes() << std::endl;
    // std::cout << "Image tensor min: " << tensor_image.min().item<float>()
    //           << ", max: " << tensor_image.max().item<float>() << std::endl;

    return tensor_image.clone(); // Return a deep copy of the tensor
}

cv::Mat infer(torch::jit::script::Module& model, torch::Tensor& image_tensor, const cv::Size& original_size) {
    model.eval(); // Ensure the model is in evaluation mode

    // Move the image tensor to GPU if available
    if (torch::cuda::is_available()) {
        image_tensor = image_tensor.to(torch::kCUDA);
        model.to(torch::kCUDA);
    } else {
        image_tensor = image_tensor.to(torch::kCPU);
        model.to(torch::kCPU);
    }

    // Inference
    torch::NoGradGuard no_grad; // Disable gradient computation
    auto output = model.forward({image_tensor});  // Perform forward pass
    torch::cuda::synchronize();  // After the forward pass

    // Access the "out" key from the output dictionary - DeepLabV3 specific
    auto out_tensor = output.toGenericDict().at("out").toTensor();  // Get the output tensor from the dict

    // std::cout << "Shape of out_tensor: " << out_tensor.sizes() << std::endl;
    // auto avg_value = out_tensor.mean().item<float>();
    // std::cout << "Average value of all pixels: " << avg_value << std::endl;

    // Get softmax probabilities and extract the apple class (index 1)
    auto probabilities = torch::softmax(out_tensor, 1);

    // std::cout << "Shape of probabilities: " << probabilities.sizes() << std::endl;

    // Extract the class probabilities for the apple class (index 1)
    auto apple_prob = probabilities[0][1].unsqueeze(0); // Take the class 1 (apples) and keep as 3D

    // Fix: Add an extra dimension to make it a 4D tensor
    apple_prob = apple_prob.unsqueeze(0); // Shape becomes [1, 1, height, width]

    // Resize the probabilities to original size
    auto resized_prob = torch::nn::functional::interpolate(
        apple_prob,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{original_size.height, original_size.width})
            .mode(torch::kBilinear)
            .align_corners(false));

    // Convert to CPU and extract the data as a cv::Mat
    resized_prob = resized_prob.squeeze().cpu(); // Remove batch and channel dimensions

    // Create cv::Mat with float data from the tensor
    cv::Mat probability_map(original_size, CV_32F, resized_prob.data_ptr<float>()); 
 
    // Convert raw probability map to 8-bit for saving (rescale first)
    cv::Mat raw_prob_8u;
    probability_map.convertTo(raw_prob_8u, CV_8U, 255.0);

    // Save the raw probability map as an image
    cv::imwrite("../raw_probability_map.png", raw_prob_8u);
    std::cout << "Saved raw probability map as raw_probability_map.png" << std::endl;

    return probability_map;
}

// Function to save and visualize the result
void plot_result(const cv::Mat& original_image, const cv::Mat& probability_map, const std::string& save_path = "") {
    // Normalize the probability map for better visualization
    cv::Mat normalized_prob_map;
    cv::normalize(probability_map, normalized_prob_map, 0, 255, cv::NORM_MINMAX);
    normalized_prob_map.convertTo(normalized_prob_map, CV_8U);

    // Apply colormap for visualization
    cv::Mat colored_prob_map;
    cv::applyColorMap(normalized_prob_map, colored_prob_map, cv::COLORMAP_VIRIDIS);

    // Overlay the probability map on the original image
    cv::Mat overlay_image;
    cv::addWeighted(original_image, 0.7, colored_prob_map, 0.3, 0, overlay_image);

    // Display results
    cv::imshow("Original Image", original_image);
    cv::imshow("Apple Probability Map", colored_prob_map);
    cv::imshow("Overlayed Image", overlay_image);
    cv::waitKey(0);

    // Optionally save the result
    if (!save_path.empty()) {
        cv::imwrite(save_path, overlay_image);
        std::cout << "Result saved to " << save_path << std::endl;
    }
}