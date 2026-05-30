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
# Step 3: Feature Matching
# ──────────────────────────────────────────

def match_features(des1, des2):
    """Match SIFT descriptors using BFMatcher + Lowe's ratio test."""
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in raw_matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"\n🔗 Total raw matches    : {len(raw_matches)}")
    print(f"✅ Good matches (Lowe's) : {len(good_matches)}")
    return good_matches


def show_matches(img1, kp1, img2, kp2, good_matches):
    """Draw the good matched keypoints between two images."""
    matched_img = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        good_matches[:50],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    matched_rgb = cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(20, 8))
    plt.imshow(matched_rgb)
    plt.title(f"Feature Matching — Top 50 of {len(good_matches)} Good Matches",
              fontsize=15, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("output/matches.png")
    print("✅ Match image saved → output/matches.png")
    plt.show()


# ──────────────────────────────────────────
# Step 4: Homography + Warping
# ──────────────────────────────────────────

def compute_homography(kp1, kp2, good_matches):
    """Compute Homography matrix using RANSAC."""
    if len(good_matches) < 4:
        raise Exception("❌ Not enough matches to compute Homography!")

    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    inliers = int(mask.sum())
    print(f"\n📐 Homography Matrix computed!")
    print(f"✅ Inliers (RANSAC): {inliers} / {len(good_matches)}")
    print(f"\nH Matrix:\n{H}")
    return H


def warp_and_stitch(img1, img2, H):
    """Warp img1 to align with img2 and stitch them together."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Corners of img1 warped into img2 space
    corners_img1 = np.float32([
        [0, 0], [0, h1], [w1, h1], [w1, 0]
    ]).reshape(-1, 1, 2)

    corners_img2 = np.float32([
        [0, 0], [0, h2], [w2, h2], [w2, 0]
    ]).reshape(-1, 1, 2)

    warped_corners = cv2.perspectiveTransform(corners_img1, H)
    all_corners = np.concatenate((corners_img2, warped_corners), axis=0)

    # Find bounding box
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel())
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel())

    # Translation matrix to shift into positive coordinates
    translation = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ], dtype=np.float64)

    output_size = (x_max - x_min, y_max - y_min)

    # Warp img1 into panorama space
    warped = cv2.warpPerspective(img1, translation @ H, output_size)

    # Paste img2 into the panorama
    warped[-y_min:h2 + (-y_min), -x_min:w2 + (-x_min)] = img2

    print(f"\n✅ Panorama size: {warped.shape}")
    return warped


def show_panorama(panorama):
    """Display and save the final panorama."""
    panorama_rgb = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(22, 8))
    plt.imshow(panorama_rgb)
    plt.title("🌄 Final Panoramic Image (SIFT)", fontsize=16, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("output/panorama.png", dpi=150)
    print("✅ Panorama saved → output/panorama.png")
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
    print(f"\n🔑 Keypoints in para11.jpg : {len(kp1)}")
    print(f"🔑 Keypoints in para12.jpg : {len(kp2)}")

    # 3. Show keypoints
    show_keypoints(img1, kp1, img2, kp2)

    # 4. Match features
    good_matches = match_features(des1, des2)

    # 5. Show matches
    show_matches(img1, kp1, img2, kp2, good_matches)

    # 6. Compute Homography
    H = compute_homography(kp1, kp2, good_matches)

    # 7. Warp and stitch
    panorama = warp_and_stitch(img1, img2, H)

    # 8. Show final panorama
    show_panorama(panorama)