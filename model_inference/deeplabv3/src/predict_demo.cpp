// ./predict_demo \
        /home/quanvu/ros/apple_ws/src/segmentation_module/deeplabv3_apples/output/2024_10_02_23_37_35/model.pt \
        /home/quanvu/uts/APPLE_DATA/few_test_images 

// ./predict_demo \
        /home/quanvu/git/segmentation-module/deeplabv3_apples/output/2024_10_02_23_37_35/model.pt \
        /media/quanvu/T7\ Shield/6_UBUNTU_20/uts/APPLE_DATA/few_test_images/

// ./predict_demo \
        /home/quanvu/git/segmentation-module/deeplabv3_apples/output/2024_10_02_23_37_35/model.pt \
        /home/quanvu/Desktop/apple_gazebo/

#include <iostream>
#include <filesystem>
#include "predict.cpp"  // Assuming you saved the previous ModelInference class in a header file

int main(int argc, const char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: ./predict_demo <path-to-exported-script-module> <image-folder>" << std::endl;
        return -1;
    }

    std::string model_path = argv[1];
    std::string image_folder = argv[2];
    // std::string output_folder = argv[3];

    std::cout << "OpenCV Version: " << CV_VERSION << std::endl;

    cv::Size input_size(800, 800);
    ModelInference model_inference(model_path, input_size);

    // // Ensure the output folder exists
    // if (!std::filesystem::exists(output_folder)) {
    //     std::filesystem::create_directory(output_folder);
    // }

    // Process each image in the folder
    int i = 0;
    for (const auto& entry : std::filesystem::directory_iterator(image_folder)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
            std::string image_path = entry.path().string();
            std::string image_name = entry.path().filename().string();
            std::cout << "----------------------------------------" << std::endl;
            std::cout << "\nProcessing: " << image_name << std::endl;

            // Load the image
            // cv::Mat original_image = cv::imread(image_path);
            // cv::Mat original_image = cv::imread("/home/quanvu/Desktop/apple_gazebo/apples_gazebo.png");
            cv::Mat original_image;


            // std::cout << "Image type (16 for CV_8UC3): " << original_image.type() << std::endl;
            // std::cout << "Image size: " << original_image.size() << std::endl;

            // if (original_image.empty()) {
            //     std::cerr << "Error loading image: " << image_path << std::endl;
            //     continue; // Skip if image loading failed
            // }

            // Perform inference
            std::cout << "Inferring image..." << std::endl;
            cv::Mat probability_map = model_inference.infer_single_image(original_image);

            if (probability_map.empty()) {
                std::cerr << "Inference failed for image: " << image_name << std::endl;
                continue;  // Skip if inference fails
            }

            double minVal, maxVal;
            cv::minMaxLoc(probability_map, &minVal, &maxVal);
            std::cout << "\nMin value in segmentation mask: " << minVal << std::endl;
            std::cout << "Max value in segmentation mask: " << maxVal << std::endl;

            // Save the result
            cv::Mat raw_prob_8u;
            probability_map.convertTo(raw_prob_8u, CV_8U, 255.0);





            cv::Mat mask_for_save;
            probability_map.convertTo(mask_for_save, CV_8U, 255.0);  // Scale to 0-255 range    
            int white_below_800 = 0;
            for (int u = 0; u < mask_for_save.cols; ++u) {
                for (int v = 0; v < mask_for_save.rows; ++v) {
                    if (v < 800){
                        // mask_for_save.at<uchar>(v, u) = 255;
                        if (mask_for_save.at<uchar>(v, u) > 230) white_below_800++;
                    } 
                        
                }
            }
            std::cout << "White pixels below 800: " << white_below_800 << std::endl;

            std::string desktop_path = "/home/quanvu/Desktop/apple_testing/";
            std::string mask_path               = desktop_path + "segmentation_mask.png";
            if (cv::imwrite(mask_path, mask_for_save)) {
                std::cout << "Current image saved to " << mask_path.c_str() << std::endl;
            } else {
                std::cout << "Failed to save segmentation mask" << std::endl;
            }





            // Save the raw probability map as an image
            std::string output_path = "../raw_probability_map_" + std::to_string(i) + ".png";
            cv::imwrite(output_path, raw_prob_8u);
            std::cout << "\nSaved raw probability map as " << output_path << std::endl;
            i++;
        }
    }

    return 0;
}
