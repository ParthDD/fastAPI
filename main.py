from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import io
from fastapi.responses import StreamingResponse
from PIL import Image

app = FastAPI()

def process_image(image_data: bytes, process_type: str):
    """Apply different image processing techniques based on process_type"""
    
    # Convert bytes to numpy array
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Apply transformations based on request
    if process_type == "gray":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif process_type == "blur":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(gray, (5, 5), 0)
    elif process_type == "canny":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        img = cv2.Canny(blur, 10, 150)

    # Convert numpy array back to image
    _, img_encoded = cv2.imencode(".png", img)
    return io.BytesIO(img_encoded.tobytes())

@app.post("/process_image/")
async def upload_file(file: UploadFile = File(...), process_type: str = "gray"):
    """
    Upload an image and apply one of the processing techniques: gray, blur, or canny.
    """
    image_bytes = await file.read()
    processed_img = process_image(image_bytes, process_type)

    return StreamingResponse(processed_img, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
