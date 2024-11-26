#include "predict.cpp"


// ./predict_demo \
        /home/quanvu/ros/apple_ws/src/segmentation_module/deeplabv3_apples/output/2024_10_02_23_37_35/model.pt \
        /home/quanvu/uts/APPLE_DATA/few_test_images \
        /media/quanvu/T7\ Shield/APPLE_OUTPUT

// ./predict_demo \
        /home/quanvu/ros/apple_ws/src/segmentation_module/deeplabv3_apples/output/2024_10_02_23_37_35/model.pt \
        /home/quanvu/uts/APPLE_DATA/few_test_images \
        /media/quanvu/T7\ Shield/APPLE_OUTPUT


int main(int argc, const char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: ./predict_demo <path-to-exported-script-module> <image-folder> <output-folder>" << std::endl;
        return -1;
    }

    std::string model_path = argv[1];
    std::string image_folder = argv[2];
    std::string output_folder = argv[3];

    // Load the model
    torch::jit::script::Module model;
    try {
        model = torch::jit::load(model_path);

        // Move the model to GPU if available
        if (torch::cuda::is_available()) {
            model.to(torch::kCUDA);
            std::cout << "CUDA is available. Moving model to GPU." << std::endl;
        } else {
            model.to(torch::kCPU);
            std::cout << "CUDA is not available. Moving model to CPU." << std::endl;
        }
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
        return -1;
    }
    std::cout << "Model loaded successfully.\n";

    // Inference parameters
    cv::Size input_size(800, 800); // Input size
    // std::filesystem::create_directory(output_folder);

    // Process each image in the folder
    for (const auto& entry : std::filesystem::directory_iterator(image_folder)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
            std::string image_path = entry.path().string();
            std::string image_name = entry.path().filename().string();
            std::cout << "\nProcessing: " << image_name << std::endl;

            // Preprocess the image
            cv::Mat original_image = cv::imread(image_path);
            cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
            std::cout <<"Preprocess image" << std::endl;
            if (image.empty()) 
                std::cerr << "Error loading image: " << image_path << std::endl;
            
            auto image_tensor = preprocess_image(image, input_size);

            if (!image_tensor.defined()) {
                continue; // Skip if image loading failed
            }

            // Inference
            std::cout <<"Inferring image" << std::endl;
            auto probability_map = infer(model, image_tensor, original_image.size());

            // Save and display results
            // std::string save_path = output_folder + "/pred_" + image_name;
            // plot_result(original_image, probability_map, save_path);
        }
    }

    return 0;
}
