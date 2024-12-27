from tensorflow.keras.applications import MobileNetV2

# Tải mô hình MobileNetV2 đã huấn luyện trên ImageNet
model = MobileNetV2(weights='imagenet')

# Lưu mô hình vào thư mục models/
model.save('models/acne_detection_model.h5')

print("Mô hình đã được lưu thành công!")
