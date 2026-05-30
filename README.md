# 🌄 Pano-Vision

Combine two partially overlapping images into a single panoramic view using the **SIFT algorithm** and OpenCV.

## 📌 How It Works
1. Detect keypoints using SIFT
2. Match features between two images
3. Compute Homography using RANSAC
4. Warp and stitch into a panorama

## 🛠️ Tech Stack
- Python
- OpenCV
- NumPy
- Matplotlib

## 📁 Project Structure
PANO-VISION/
├── images/        ← input images
├── output/        ← stitched result saved here
├── src/           ← source code
│   └── stitch.py
└── requirements.txt

## 🚀 Run
```bash
pip install -r requirements.txt
python src/stitch.py
```