import os
import cv2
import numpy as np
import aiofiles
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import threading
from pyngrok import ngrok
from ultralytics import YOLO
from dtbase import db, upload_to_imgbb, upload_to_streamable
from datetime import datetime
import gdown
import uvicorn

app = FastAPI()

NGROK_AUTH_TOKEN = "2tcouva4KHG2fccLtZPW7PDXMvZ_4YCgrCFDUKea2cJUhYj8t"
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# ✅ Đường dẫn model và ID Google Drive
MODEL_PATH = "best.pt"
DRIVE_ID = "1GNc8GNxEhlU4f2gOHvVpIWLqbDIjgU2R"

# ✅ Kiểm tra và tải model nếu chưa tồn tại
if not os.path.exists(MODEL_PATH):
    print(f"📥 Đang tải mô hình từ Google Drive ID: {DRIVE_ID}...")
    gdown.download(f"https://drive.google.com/uc?id={DRIVE_ID}", MODEL_PATH, quiet=False)
    print("✅ Tải mô hình thành công!")

print(f"🔄 Đang tải mô hình YOLOv8 từ {MODEL_PATH}...")
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
            <a href="/start-camera/">Start Camera</a><br>
            <a href="/stop-camera/">Stop Camera</a>
        </body>
    </html>
    """

@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if image is None:
            return {"error": "Không thể giải mã hình ảnh."}

        results = model(image)
        predictions, person_count = [], 0

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
            "person_count": person_count
        })

        return {"person_count": person_count, "image_url": imgbb_url}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict-video/")
async def predict_video(file: UploadFile = File(...)):
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
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_count, total_swimmer_count, prev_swimmer_count = 0, 0, 0
    prev_counts = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 5 == 0:
            results = model(frame)
            if results:
                frame = results[0].plot()
                current_swimmer_count = sum(
                    1 for box in results[0].boxes if model.names[int(box.cls.item())] == "swimmer"
                )
                prev_counts.append(current_swimmer_count)
                if len(prev_counts) == 3:
                    avg_swimmer_count = round(sum(prev_counts) / 3)
                    if avg_swimmer_count != prev_swimmer_count:
                        total_swimmer_count = avg_swimmer_count
                        prev_swimmer_count = avg_swimmer_count
                    prev_counts = []

        out.write(frame)

    cap.release()
    out.release()

    streamable_url = upload_to_streamable(output_video_path)
    if os.path.exists(input_video_path): os.remove(input_video_path)
    if os.path.exists(output_video_path): os.remove(output_video_path)

    db.predictions.insert_one({
        "video_url": streamable_url,
        "total_swimmer_count": total_swimmer_count
    })

    return {"total_swimmer_count": total_swimmer_count, "video_url": streamable_url}

recording = False
camera_thread = None

@app.get("/start-camera/")
def start_camera():
    global recording, camera_thread
    if recording:
        return {"message": "Camera đang chạy!"}

    recording = True

    def camera_worker():
        global recording
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Không thể mở webcam.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"camera_output_{timestamp}.mp4"
        out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (640, 480))
        total_swimmer_count = 0

        while recording:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            frame = results[0].plot()
            swimmer_count = sum(
                1 for box in results[0].boxes if model.names[int(box.cls.item())].lower() == "swimmer"
            )
            total_swimmer_count = swimmer_count

            out.write(frame)
            cv2.imshow("🟢 Realtime Detection (Nhấn 'q' để dừng)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                recording = False
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        streamable_url = upload_to_streamable(output_filename)
        if os.path.exists(output_filename):
            os.remove(output_filename)

        db.predictions.insert_one({
            "video_url": streamable_url,
            "total_swimmer_count": total_swimmer_count,
            "source": "realtime",
            "timestamp": timestamp
        })

        print(f"✅ Video {output_filename} đã được upload!")

    camera_thread = threading.Thread(target=camera_worker)
    camera_thread.start()
    return {"message": "✅ Camera đang ghi hình và phân tích realtime."}

@app.get("/stop-camera/")
def stop_camera():
    global recording
    recording = False
    return {"message": "⏹ Camera đã được tắt và video đã được xử lý."}


def start_ngrok():
    public_url = ngrok.connect(8000).public_url
    print(f"🔥 Ngrok URL: {public_url}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

