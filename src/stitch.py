import cv2
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────
# Step 2: Load Images & Detect SIFT Keypoints
# ──────────────────────────────────────────

def load_images(path1, path2):
    """Load two images from given paths."""
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    if img1 is None or img2 is None:
        raise FileNotFoundError("❌ Images not found! Check the images/ folder.")

    print(f"✅ Loaded para11.jpg → {img1.shape}")
    print(f"✅ Loaded para12.jpg → {img2.shape}")
    return img1, img2


def detect_keypoints(img):
    """Convert to grayscale and detect SIFT keypoints."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


def show_keypoints(img1, kp1, img2, kp2):
    """Visualize keypoints on both images side by side."""
    img1_kp = cv2.drawKeypoints(
        img1, kp1, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    img2_kp = cv2.drawKeypoints(
        img2, kp2, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Convert BGR → RGB for matplotlib
    img1_rgb = cv2.cvtColor(img1_kp, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2_kp, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].imshow(img1_rgb)
    axes[0].set_title(f"para11.jpg — {len(kp1)} Keypoints", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(img2_rgb)
    axes[1].set_title(f"para12.jpg — {len(kp2)} Keypoints", fontsize=13)
    axes[1].axis("off")

    plt.suptitle("SIFT Keypoint Detection", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("output/keypoints.png")
    print("✅ Keypoints image saved → output/keypoints.png")
    plt.show()


# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────
if __name__ == "__main__":

    # 1. Load images
    img1, img2 = load_images("images/para11.jpg", "images/para12.jpg")

    # 2. Detect keypoints
    kp1, des1 = detect_keypoints(img1)
    kp2, des2 = detect_keypoints(img2)

    print(f"\n🔑 Keypoints found in para11.jpg: {len(kp1)}")
    print(f"🔑 Keypoints found in para12.jpg: {len(kp2)}")

    # 3. Show keypoints
    show_keypoints(img1, kp1, img2, kp2)