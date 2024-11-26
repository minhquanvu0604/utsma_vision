import os
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Real-Time Inference')

from PIL import Image

import torch
from torchvision import transforms

from deeplabv3_apples.predict import load_model, preprocess_image, infer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RealTimeInference:
    """
    Real-time inference class for semantic segmentation.
    
    Args:
        model_path (str): Path to the model file.
        input_size (tuple): Size of the input image (width, height).
        num_classes (int): Number of classes in the model.
    
    @TODO: config file
    """
    def __init__(self, model_path, input_size, num_classes=2):
        """
        Initializes the real-time inference class by loading the model and setting up the input size.
        """
        self.model = load_model(model_path, num_classes)
        self.input_size = input_size
        self.model.eval()  # Set the model to evaluation mode
        logger.info("Real-time inference initialized.")

    def infer_single_image(self, image):
        """
        Performs inference on a single image and returns the predicted mask.
        """
        image_tensor, original_image = preprocess_image(image)
        if image_tensor is None:
            return None  # Return if there's an error

        original_size = original_image.size
        predicted_mask = infer(self.model, image_tensor, original_size)

        return predicted_mask  # Return the probability mask
