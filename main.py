import os
import cv2
import numpy as np
import aiofiles
from fastapi import FastAPI, UploadFile, File
import uvicorn
from pyngrok import ngrok
from ultralytics import YOLO
from dtbase import db, upload_to_imgbb, upload_to_streamable # Thay đổi sang ImgBB
from collections import deque  # 🔥 Lưu lịch sử số lượng swimmer
from datetime import datetime
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi import Request
import gdown

# ✅ Khởi tạo FastAPI
app = FastAPI()


# Đường dẫn lưu file
MODEL_PATH = "best.pt"

# ID của file Google Drive
FILE_ID = "1GNc8GNxEhlU4f2gOHvVpIWLqbDIjgU2R"
GDRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Tải mô hình nếu chưa tồn tại
if not os.path.exists(MODEL_PATH):
    print("⬇️ Đang tải mô hình YOLOv8 từ Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    print("✅ Tải mô hình thành công!")

# Load mô hình
print("🔄 Đang tải mô hình YOLOv8...")
model = YOLO(MODEL_PATH)
print("✅ Mô hình YOLOv8 đã sẵn sàng!")

@app.get("/", response_class=HTMLResponse)
async def homepage():
    return """
    <html>
        <head>
            <title>YOLOv8 Swimmer Detection</title>
        </head>
        <body>
            <h2>Upload Image</h2>
            <form action="/predict-image/" enctype="multipart/form-data" method="post">
                <input type="file" name="file">
                <input type="submit">
            </form>

            <h2>Upload Video</h2>
            <form action="/predict-video/" enctype="multipart/form-data" method="post">
                <input type="file" name="file">
                <input type="submit">
            </form>

            <h2>Realtime Camera Detection</h2>
            <a href="/camera/">Camera</a><br>
            <h2>Lịch sử dự đoán</h2>
            <a href="/history">🔍 Xem lịch sử</a>
        </body>
    </html>
    """
@app.get("/camera", response_class=HTMLResponse)
async def camera_interface():
    video_url = getattr(app.state, "last_camera_url", None)
    video_html = f"""
        <h3>🎞️ Video mới nhất</h3>
        <video width="640" controls>
            <source src="{video_url}" type="video/mp4">
            Trình duyệt không hỗ trợ phát video.
        </video>
    """ if video_url else "<p>Chưa có video nào.</p>"

    return f"""
    <html>
        <head><title>Realtime Camera Detection</title></head>
        <body>
            <h2>📸 Realtime Detection</h2>
            <form action="/start-camera/" method="get">
                <button type="submit">▶️ Bắt đầu Camera</button>
            </form>
            <form action="/stop-camera/" method="get">
                <button type="submit">⏹ Dừng Camera</button>
            </form>
            <br>
            {video_html}
            <br>
            <a href="/">⬅ Quay lại trang chính</a>
        </body>
    </html>
    """


@app.get("/history", response_class=HTMLResponse)
async def view_history():
    rows = ""
    for item in db.predictions.find().sort("timestamp", -1).limit(10):
        count = item.get("total_swimmer_count") or item.get("person_count") or 0
        time = item.get("timestamp", "Không có")
        url = item.get("video_url") or item.get("image_url") or "#"
        media_type = "Hình ảnh" if "image_url" in item else "Video"
        rows += f"""
        <tr>
            <td>{time}</td>
            <td>{count}</td>
            <td>{media_type}</td>
            <td><a href="{url}" target="_blank">Xem</a></td>
        </tr>
        """ 

    return f"""
    <html>
        <head><title>Lịch sử dự đoán</title></head>
        <body>
            <h2>Lịch sử dự đoán gần đây</h2>
            <table border="1">
                <tr>
                    <th>Thời gian</th>

                    <th>Số lượng swimmer</th>
                    <th>Loại</th>
                    <th>Xem</th>
                </tr>
                {rows}
            </table>
            <br>
            <a href="/">⬅️ Quay lại trang chính</a>
        </body>
    </html>
    """

@app.post("/predict-image/", response_class=HTMLResponse)
async def predict_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        
        if image is None:
            return HTMLResponse(content="Không thể giải mã ảnh.", status_code=400)
        
        results = model(image)
        predictions = []
        person_count = 0
        
        for result in results:
            for i in range(len(result.boxes)):
                x1, y1, x2, y2 = map(int, result.boxes.xyxy[i].tolist())
                score = round(result.boxes.conf[i].item(), 2)
                label = model.names[int(result.boxes.cls[i].item())]

                if label.lower() == "swimmer":
                    person_count += 1

                predictions.append({
                    "label": label,
                    "confidence": score,
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                })
        
        image_with_boxes = results[0].plot()
        output_path = "output.jpg"
        cv2.imwrite(output_path, image_with_boxes)

        imgbb_url = upload_to_imgbb(output_path)

        if os.path.exists(output_path):
            os.remove(output_path)
        
        db.predictions.insert_one({
            "image_url": imgbb_url,
            "predictions": predictions,
            "person_count": person_count,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")

        })
        
        html_content = f"""
        <html>
            <body>
                <h3>✅ Dự đoán thành công!</h3>
                <p>Số swimmer phát hiện: <strong>{person_count}</strong></p>
                <img src="{imgbb_url}" alt="Predicted Image" width="640">
                <br><br><a href="/">⬅ Quay lại trang chủ</a>
            </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    
    except Exception as e:
        return HTMLResponse(content=f"Lỗi xử lý ảnh: {e}", status_code=500)

    

# ✅ XỬ LÝ VIDEO & LƯU VÀO MONGODB + STREAMABLE
@app.post("/predict-video/")
async def predict_video(file: UploadFile = File(...)):
    """Nhận video, xử lý bằng YOLOv8, lưu MongoDB & Streamable"""
    input_video_path = "temp_input.mp4"
    async with aiofiles.open(input_video_path, "wb") as f:
        await f.write(await file.read())

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        return {"error": "Không thể mở video!"}

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    total_swimmer_count = 0
    prev_swimmer_count = 0  # Biến theo dõi số swimmer nhóm trước
    prev_counts = []  # Lưu số swimmer của nhóm 3 frame gần nhất

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Hết video

        frame_count += 1

        # ✅ Chỉ xử lý mỗi 5 frame để tối ưu
        if frame_count % 5 == 0:
            results = model(frame)

            if results and len(results) > 0:
                frame = results[0].plot()  # Vẽ bounding box lên frame

                # ✅ Đếm số swimmer trong frame hiện tại
                current_swimmer_count = sum(
                    1 for box in results[0].boxes if model.names[int(box.cls.item())] == "swimmer"
                )
                prev_counts.append(current_swimmer_count)

                # ✅ Khi đủ 3 frame, cập nhật số swimmer nếu có thay đổi
                if len(prev_counts) == 3:
                    avg_swimmer_count = round(sum(prev_counts) / 3)  # Lấy trung bình nhóm 3 frame

                    if avg_swimmer_count != prev_swimmer_count:
                        total_swimmer_count = avg_swimmer_count  # Cập nhật số swimmer
                        prev_swimmer_count = avg_swimmer_count  # Lưu lại để so sánh với nhóm tiếp theo

                    prev_counts = []  # ✅ Reset nhóm 3 frame để tiếp tục theo dõi

        out.write(frame)  # ✅ Ghi frame vào video đầu ra

    # ✅ Giải phóng tài nguyên
    cap.release()
    out.release()

    # ✅ Upload video lên Streamable
    streamable_url = upload_to_streamable(output_video_path)

    # ✅ Xóa các file tạm
    for file_path in [input_video_path, output_video_path]:
        if os.path.exists(file_path):
            os.remove(file_path)

    # ✅ Lưu vào MongoDB
    db.predictions.insert_one({
        "video_url": streamable_url,
        "total_swimmer_count": total_swimmer_count, # 🔥 Tổng số swimmer sau khi xử lý
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")

    })

    html = f"""
    <html>
        <body>
            <h3>✅ Video đã xử lý thành công!</h3>
            <p>Tổng số swimmer: <strong>{total_swimmer_count}</strong></p>
            <p>👉 <a href="{streamable_url}" target="_blank">Xem video trên Streamable</a></p>
            <iframe src="{streamable_url}" width="640" height="360" frameborder="0" allowfullscreen></iframe>
            <br><br><a href="/">⬅ Quay lại trang chủ</a>
        </body>
    </html>
    """
    return HTMLResponse(content=html)



