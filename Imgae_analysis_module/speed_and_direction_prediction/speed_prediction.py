from utils.image_utils import image_saver

is_vehicle_detected = [0]
current_frame_number_list = [0]
current_frame_number_list_2 = [0]
bottom_position_of_detected_vehicle = [0]


def predict_speed(
    speed_factor,
    top,
    bottom,
    right,
    left,
    current_frame_number,
    crop_img,
    roi_position,
    NUM
    ):
    speed = 'n.a.'  #initialization
    direction = 'n.a.'  #initialization
    scale_constant = 1  # Manual calibration, without camera calibration
    isInROI = True  # Is it in the ROI area
    update_csv = False

    if bottom < roi_position+50:
        scale_constant = 1  #manual scaling
    elif bottom > (roi_position+50) and bottom < (roi_position+70):
        scale_constant = 2  #manual scaling
    else:
        isInROI = False

    if len(bottom_position_of_detected_vehicle) != 0 and bottom \
        - bottom_position_of_detected_vehicle[0] > 0 and (roi_position-5) \
        < bottom_position_of_detected_vehicle[0] \
        and bottom_position_of_detected_vehicle[0] < (roi_position+40) \
        and roi_position < bottom+100 and (current_frame_number - current_frame_number_list_2[0])>24:
        is_vehicle_detected.insert(0, 1)#Insert 1 into is_vehicle_detected [1,0], if there is no vehicle then is_vehicle_detected =[0]
        update_csv = True
        image_saver.save_image(crop_img, NUM)  # Save the vehicle image to a local file that the UI can read
        current_frame_number_list_2.insert(0, current_frame_number)
    # print("bottom_position_of_detected_vehicle[0]: " + str(bottom_position_of_detected_vehicle[0]))
    # print("bottom: " + str(bottom))
    if bottom > bottom_position_of_detected_vehicle[0]:
        direction = 'down'
    else:
        direction = 'up'

    if isInROI:
        pixel_length = bottom - bottom_position_of_detected_vehicle[0]
        scale_real_length = pixel_length * speed_factor  # multiplied by speed_factor to convert pixel length to real length in meters
        total_time_passed = current_frame_number - current_frame_number_list[0]
        scale_real_time_passed = total_time_passed * 24  # get the elapsed total time for a vehicle to pass through ROI area (24 = fps)
        if scale_real_time_passed != 0:
            speed = scale_real_length / scale_real_time_passed / scale_constant  # performing manual scaling because we have not performed camera calibration
            speed = speed / 6 * 40  # use reference constant to get vehicle speed prediction in kilometer unit
            current_frame_number_list.insert(0, current_frame_number)
            bottom_position_of_detected_vehicle.insert(0, bottom)
    return (direction, speed, is_vehicle_detected, update_csv)
