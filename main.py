import cv2
import mediapipe as mp
import numpy as np
import os
import tensorflow.lite as tflite
import pygame

# Khởi tạo pygame để phát âm thanh (mình dùng pygame để phát file mp3)
pygame.mixer.init()

# Tạo thư mục eye_images để lưu ảnh mắt tạm thời (nếu chưa có thì tạo mới)
output_dir = "eye_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Khởi tạo các module từ Mediapipe (mình dùng Mediapipe để phát hiện khuôn mặt)
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Đường dẫn đến file mô hình face_landmarker.task (file này để phát hiện khuôn mặt)
model_path = os.path.join("assets", "face_landmarker.task")

# Đường dẫn đến file mô hình eye_classifier.tflite (file này để phân loại mắt mở hay nhắm)
eye_classifier_path = os.path.join("assets", "eye_classifier.tflite")

# Đường dẫn đến file âm thanh alarm.mp3 (file này để phát âm thanh cảnh báo khi buồn ngủ)
alarm_path = os.path.join("assets", "alarm.mp3")

# Tải file âm thanh vào để phát (mình dùng file alarm.mp3 để cảnh báo)
alarm_sound = pygame.mixer.Sound(alarm_path)

# Thiết lập Face Landmarker để chạy ở chế độ video (mình chọn chế độ video vì dùng webcam)
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO
)

# Khởi tạo Face Landmarker (để bắt đầu dùng Mediapipe phát hiện khuôn mặt)
landmarker = FaceLandmarker.create_from_options(options)

# Tải mô hình eye_classifier.tflite (mô hình này mình dùng để phân loại mắt)
interpreter = tflite.Interpreter(model_path=eye_classifier_path)
interpreter.allocate_tensors()

# Lấy thông tin đầu vào và đầu ra của mô hình (để biết cách đưa dữ liệu vào và lấy kết quả ra)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Mở webcam (mình dùng webcam mặc định, số 0 là webcam của máy)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không thể mở webcam")  # Nếu không mở được webcam thì báo lỗi
    exit()

frame_count = 0  # Biến để đếm số khung hình (dùng để đặt tên file ảnh mắt)
closed_eye_frame_count = 0  # Biến để đếm số khung hình mà cả hai mắt đều nhắm
DROWSINESS_THRESHOLD = 8  # Ngưỡng để phát hiện buồn ngủ (8 khung hình, khoảng 0,133 giây với webcam 60 FPS)
is_drowsy = False  # Biến để kiểm tra xem tài xế có đang buồn ngủ không (dùng để phát âm thanh)

while True:
    ret, frame = cap.read()  # Đọc khung hình từ webcam
    if not ret:
        print("Không thể đọc khung hình từ webcam")  # Nếu không đọc được thì báo lỗi
        break

    # Tạo bản sao của khung hình gốc (mình sao chép để không làm thay đổi khung hình gốc)
    frame_original = frame.copy()

    # Chuyển khung hình sang định dạng Mediapipe Image (Mediapipe yêu cầu định dạng này để xử lý)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Gửi khung hình để Mediapipe xử lý (mình dùng chế độ VIDEO để xử lý liên tục)
    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    # Khởi tạo trạng thái mắt (mặc định là "unknown" nếu không phát hiện được mắt)
    left_eye_status = "unknown"
    right_eye_status = "unknown"

    if result.face_landmarks:  # Nếu Mediapipe phát hiện được khuôn mặt
        for face_landmarks in result.face_landmarks:
            # Các chỉ số landmark cho mắt trái (mình lấy từ tài liệu Mediapipe)
            left_eye_indices = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144]
            # Các chỉ số landmark cho mắt phải (cũng lấy từ tài liệu Mediapipe)
            right_eye_indices = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373]

            # Tính vùng bao quanh mắt trái (tìm tọa độ min/max để cắt vùng mắt)
            min_x_left, max_x_left = float('inf'), float('-inf')
            min_y_left, max_y_left = float('inf'), float('-inf')
            for idx in left_eye_indices:
                landmark = face_landmarks[idx]
                x = landmark.x * frame.shape[1]  # Chuyển tọa độ chuẩn hóa sang pixel (theo chiều ngang)
                y = landmark.y * frame.shape[0]  # Chuyển tọa độ chuẩn hóa sang pixel (theo chiều dọc)
                min_x_left = min(min_x_left, x)
                max_x_left = max(max_x_left, x)
                min_y_left = min(min_y_left, y)
                max_y_left = max(max_y_left, y)

            # Tính vùng bao quanh mắt phải (tương tự như mắt trái)
            min_x_right, max_x_right = float('inf'), float('-inf')
            min_y_right, max_y_right = float('inf'), float('-inf')
            for idx in right_eye_indices:
                landmark = face_landmarks[idx]
                x = landmark.x * frame.shape[1]
                y = landmark.y * frame.shape[0]
                min_x_right = min(min_x_right, x)
                max_x_right = max(max_x_right, x)
                min_y_right = min(min_y_right, y)
                max_y_right = max(max_y_right, y)

            # Thêm padding để vùng cắt bao quát mắt (mình thêm 10 pixel để vùng cắt rộng hơn một chút)
            padding = 10

            # Điều chỉnh vùng bao quanh mắt trái thành hình vuông (để vùng cắt không bị méo)
            width_left = max_x_left - min_x_left
            height_left = max_y_left - min_y_left
            max_side_left = max(width_left, height_left)  # Lấy cạnh lớn nhất để làm hình vuông
            center_x_left = (min_x_left + max_x_left) / 2
            center_y_left = (min_y_left + max_y_left) / 2
            half_side_left = (max_side_left / 2) + padding  # Thêm padding vào cạnh
            left_eye_rect = (
                int(max(0, center_x_left - half_side_left)),
                int(max(0, center_y_left - half_side_left)),
                int(min(frame.shape[1], center_x_left + half_side_left)),
                int(min(frame.shape[0], center_y_left + half_side_left))
            )

            # Điều chỉnh vùng bao quanh mắt phải thành hình vuông (tương tự mắt trái)
            width_right = max_x_right - min_x_right
            height_right = max_y_right - min_y_right
            max_side_right = max(width_right, height_right)
            center_x_right = (min_x_right + max_x_right) / 2
            center_y_right = (min_y_right + max_y_right) / 2
            half_side_right = (max_side_right / 2) + padding
            right_eye_rect = (
                int(max(0, center_x_right - half_side_right)),
                int(max(0, center_y_right - half_side_right)),
                int(min(frame.shape[1], center_x_right + half_side_right)),
                int(min(frame.shape[0], center_y_right + half_side_right))
            )

            # Cắt vùng mắt trái và mắt phải từ khung hình gốc
            left_eye_img = frame_original[left_eye_rect[1]:left_eye_rect[3], left_eye_rect[0]:left_eye_rect[2]]
            right_eye_img = frame_original[right_eye_rect[1]:right_eye_rect[3], right_eye_rect[0]:right_eye_rect[2]]

            # Đường dẫn để lưu ảnh mắt tạm thời (mình đặt tên theo số khung hình)
            left_eye_path = os.path.join(output_dir, f"left_eye_{frame_count}.png")
            right_eye_path = os.path.join(output_dir, f"right_eye_{frame_count}.png")

            # Resize ảnh mắt về 128x128 và chuyển sang ảnh xám (grayscale) để đưa vào mô hình
            if left_eye_img.size > 0:
                left_eye_img = cv2.resize(left_eye_img, (128, 128), interpolation=cv2.INTER_AREA)
                left_eye_img = cv2.cvtColor(left_eye_img, cv2.COLOR_BGR2GRAY)  # Chuyển sang ảnh xám
                cv2.imwrite(left_eye_path, left_eye_img)
            if right_eye_img.size > 0:
                right_eye_img = cv2.resize(right_eye_img, (128, 128), interpolation=cv2.INTER_AREA)
                right_eye_img = cv2.cvtColor(right_eye_img, cv2.COLOR_BGR2GRAY)  # Chuyển sang ảnh xám
                cv2.imwrite(right_eye_path, right_eye_img)

            # Phân loại trạng thái mắt trái bằng mô hình eye_classifier.tflite
            if left_eye_img.size > 0:
                # Chuẩn bị dữ liệu đầu vào (mình chuẩn hóa ảnh về [0, 1] để đưa vào mô hình)
                input_data = left_eye_img.astype(np.float32) / 255.0
                input_data = np.expand_dims(input_data, axis=0)  # Thêm chiều batch
                input_data = np.expand_dims(input_data, axis=-1)  # Thêm chiều kênh (vì ảnh xám là 1 kênh)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])
                # Phân loại trạng thái mắt: closed_eye (index 1), open_eye (index 0)
                left_eye_status = "open" if output_data[0][1] > output_data[0][0] else "closed"
                # Xóa ảnh mắt trái sau khi phân loại (để không lưu lại)
                os.remove(left_eye_path)

            # Phân loại trạng thái mắt phải (tương tự mắt trái)
            if right_eye_img.size > 0:
                input_data = right_eye_img.astype(np.float32) / 255.0
                input_data = np.expand_dims(input_data, axis=0)
                input_data = np.expand_dims(input_data, axis=-1)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])
                right_eye_status = "open" if output_data[0][1] > output_data[0][0] else "closed"
                # Xóa ảnh mắt phải sau khi phân loại
                os.remove(right_eye_path)

            # Đếm số khung hình mà cả hai mắt đều nhắm (để phát hiện buồn ngủ)
            if left_eye_status == "closed" and right_eye_status == "closed":
                closed_eye_frame_count += 1
                if closed_eye_frame_count >= DROWSINESS_THRESHOLD and not is_drowsy:
                    # Thêm độ trễ 0,2 giây trước khi phát âm thanh (mình để 200ms cho nhanh)
                    pygame.time.delay(200)
                    is_drowsy = True
                    alarm_sound.play(-1)  # Phát âm thanh cảnh báo liên tục
            else:
                closed_eye_frame_count = 0  # Reset nếu một trong hai mắt mở
                if is_drowsy:
                    is_drowsy = False
                    alarm_sound.stop()  # Dừng âm thanh khi không còn buồn ngủ

            # Hiển thị trạng thái mắt trái và mắt phải trên khung hình (mình để màu xanh lá cho dễ nhìn)
            cv2.putText(frame, f"Left Eye: {left_eye_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Right Eye: {right_eye_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Nếu phát hiện buồn ngủ thì hiển thị cảnh báo (mình để màu đỏ để nổi bật)
            if closed_eye_frame_count >= DROWSINESS_THRESHOLD:
                cv2.putText(frame, "Drowsiness Detected!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Vẽ các điểm landmark lên khung hình (mình vẽ các chấm đỏ để kiểm tra Mediapipe có hoạt động đúng không)
            for landmark in face_landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    # Hiển thị khung hình lên màn hình (mình dùng OpenCV để hiện khung hình webcam)
    cv2.imshow("Face Landmarks", frame)
    frame_count += 1

    # Thoát chương trình nếu nhấn phím 'q' (mình để phím 'q' để dừng chương trình)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên (đóng webcam và các thứ khác để không bị treo máy)
cap.release()
cv2.destroyAllWindows()
landmarker.close()
pygame.mixer.quit()
