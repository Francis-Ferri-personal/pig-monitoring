from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import os

app = FastAPI()

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a directory to store uploaded and processed videos temporarily
os.makedirs("data/videos", exist_ok=True)

# Endpoint to handle video upload
@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    # Save the uploaded file
    file_location = f"data/videos/{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)

    # TODO: Here you would call your actual models to process the video.
    # For now, we simulate processing by returning the same video URL for all 3 views.
    # We will serve these files statically.

    return {
        "filename": file.filename,
        "original_url": f"/videos/{file.filename}",
        "keypoints_url": f"/videos/{file.filename}", # Replace with actual keypoints video
        "behavior_url": f"/videos/{file.filename}"  # Replace with actual behavior video
    }

# Serve the static video files
app.mount("/videos", StaticFiles(directory="data/videos"), name="videos")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
