import cv2
import torch
import numpy as np
from pyzed import sl

def main():
    # initialize ZED camera
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.CENTIMETER

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Error opening ZED camera. Exiting...")
        exit(-1)

    runtime = sl.RuntimeParameters()
    mat_l = sl.Mat()
    mat_r = sl.Mat()
    depth_map = sl.Mat()

    # load YOLOv5 model
    # model = hubconf.custom('yolov5s.pt')
    model = torch.hub.load('./', 'custom', './yolov5s.pt', source='local')

    # loop images capture and perform object detection
    while True:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            # obtain left and right images
            zed.retrieve_image(mat_l, sl.VIEW.LEFT)
            zed.retrieve_image(mat_r, sl.VIEW.RIGHT)

            # input the left image into the YOLOv5 model for object detection
            img_l = cv2.cvtColor(mat_l.get_data(), cv2.COLOR_RGBA2RGB)
            results = model(img_l, size=640)

            # obtain depth map
            zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)

            # traverse detected targets and draw bounding boxes
            for *xyxy, conf, cls in results.xyxy[0]:
                x_center = int((xyxy[0] + xyxy[2]) / 2)
                y_center = int((xyxy[1] + xyxy[3]) / 2)

                # obtain depth information of the object's center point
                depth_val = depth_map.get_value(x_center, y_center)[1]

                # draw bounding boxes and depth information
                # label = f"{depth_val}m"
                label = "%.0fcm" % depth_val
                label1 = "%.2f" % float(conf)
                cv2.rectangle(img_l, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                cv2.putText(img_l, label, (int(xyxy[0]), int(xyxy[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(img_l, results.names[int(cls)], (int(xyxy[0]), int(xyxy[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(img_l, label1, (int(xyxy[0]) + 60, int(xyxy[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                print(int(cls))

            # display Results
            cv2.imshow("Object Detection and Depth Estimation", img_l)
            key = cv2.waitKey(1)

            if key == ord('q'):
                break

    # release resources and close windows
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()