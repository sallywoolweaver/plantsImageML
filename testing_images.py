from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train/weights/best.pt")
img_path= "/home/compsci/Downloads/rasppiupload-1-001/rasppiupload/image(1154).jpg"
results = model(img_path, imgsz=640)

res = results[0]
img = res.plot()
cv2.imshow("Detections", img)
cv2.waitKey(0)
cv2.destroyAllWindows()