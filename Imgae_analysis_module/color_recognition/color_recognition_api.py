from Imgae_analysis_module.color_recognition import color_histogram_feature_extraction
from Imgae_analysis_module.color_recognition import knn_classifier
import os
from utils.image_utils import crop_image

current_path = os.getcwd()


def color_recognition(crop_img):

    (height, width, channels) = crop_img.shape
    crop_img = crop_image.crop_center(crop_img, 50, 50)  # An image is extracted from the center of the identified vehicle image, and the image is used for color recognition

    # cv2.imwrite(current_path + "/debug_utility"+".png",crop_img) # save image piece for debugging
    open(current_path + '/Imgae_analysis_module/color_recognition/' + 'test.data', 'w')
    color_histogram_feature_extraction.color_histogram_of_test_image(crop_img)  # send image piece to regonize vehicle color
    prediction = knn_classifier.main(current_path
            + '/Imgae_analysis_module/color_recognition/' + 'training.data',
            current_path + '/Imgae_analysis_module/color_recognition/'
            + 'test.data')

    return prediction
