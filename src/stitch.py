import cv2
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────
# Step 2: Load & Detect
# ──────────────────────────────────────────

def load_images(path1, path2):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    if img1 is None or img2 is None:
        raise FileNotFoundError("❌ Images not found!")
    print(f"✅ Loaded para11.jpg → {img1.shape}")
    print(f"✅ Loaded para12.jpg → {img2.shape}")
    return img1, img2


def detect_keypoints(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


def show_keypoints(img1, kp1, img2, kp2):
    img1_kp = cv2.drawKeypoints(img1, kp1, None,
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_kp = cv2.drawKeypoints(img2, kp2, None,
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
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
    print("✅ Keypoints saved → output/keypoints.png")
    plt.show()


# ──────────────────────────────────────────
# Step 3: Feature Matching
# ──────────────────────────────────────────

def match_features(des1, des2):
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
    matched_img = cv2.drawMatches(
        img1, kp1, img2, kp2, good_matches[:50], None,
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
    print("✅ Matches saved → output/matches.png")
    plt.show()


# ──────────────────────────────────────────
# Step 4: Homography
# ──────────────────────────────────────────

def compute_homography(kp1, kp2, good_matches):
    if len(good_matches) < 4:
        raise Exception("❌ Not enough matches!")

    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    inliers = int(mask.sum())
    print(f"\n📐 Homography computed!")
    print(f"✅ Inliers (RANSAC): {inliers} / {len(good_matches)}")
    print(f"\nH Matrix:\n{H}")
    return H


# ──────────────────────────────────────────
# Step 5: Warp + Gradient Blend
# ──────────────────────────────────────────

def warp_and_stitch(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners_img1 = np.float32(
        [[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    corners_img2 = np.float32(
        [[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)

    warped_corners  = cv2.perspectiveTransform(corners_img1, H)
    all_corners     = np.concatenate((corners_img2, warped_corners), axis=0)

    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel())
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel())

    translation = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ], dtype=np.float64)

    output_size = (x_max - x_min, y_max - y_min)
    W, H_out = output_size

    # ── Warp img1 ──
    warped1 = cv2.warpPerspective(img1, translation @ H, output_size)

    # ── Mask of warped img1 ──
    mask1 = cv2.warpPerspective(
        np.ones((h1, w1), dtype=np.float32),
        translation @ H, output_size
    )

    # ── Place img2 ──
    y_off, x_off = -y_min, -x_min
    canvas = np.zeros((H_out, W, 3), dtype=np.uint8)
    canvas[y_off:y_off+h2, x_off:x_off+w2] = img2

    mask2 = np.zeros((H_out, W), dtype=np.float32)
    mask2[y_off:y_off+h2, x_off:x_off+w2] = 1.0

    # ── Gradient blend in overlap zone ──
    overlap = (mask1 > 0) & (mask2 > 0)

    # Build horizontal gradient weights across overlap
    overlap_cols = np.where(overlap.any(axis=0))[0]
    blend = np.zeros((H_out, W), dtype=np.float32)

    if len(overlap_cols) > 0:
        col_start = overlap_cols[0]
        col_end   = overlap_cols[-1]
        for col in range(col_start, col_end + 1):
            t = (col - col_start) / max(col_end - col_start, 1)
            blend[:, col] = t   # 0 = use warped1 fully, 1 = use canvas fully

    # Final composite
    result = np.zeros_like(canvas, dtype=np.float32)
    for c in range(3):
        w1_ch  = warped1[:, :, c].astype(np.float32)
        w2_ch  = canvas[:, :, c].astype(np.float32)

        # In overlap: gradient blend
        blended = w1_ch * (1 - blend) + w2_ch * blend

        # Only warped1 (no img2)
        only1 = (mask1 > 0) & ~overlap
        # Only img2 (no warped1)
        only2 = (mask2 > 0) & ~overlap

        result[:, :, c] = np.where(overlap, blended,
                          np.where(only1, w1_ch,
                          np.where(only2, w2_ch, 0)))

    result = np.clip(result, 0, 255).astype(np.uint8)
    print(f"\n✅ Stitched size: {result.shape}")
    return result


# ──────────────────────────────────────────
# Step 6: Crop Black Borders
# ──────────────────────────────────────────

def crop_black_borders(img):
    """Crop black borders using largest inner rectangle."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)

    # Find the largest valid rectangle row by row
    rows = np.any(thresh > 0, axis=1)
    cols = np.any(thresh > 0, axis=0)

    row_start = np.argmax(rows)
    row_end   = len(rows) - np.argmax(rows[::-1])
    col_start = np.argmax(cols)
    col_end   = len(cols) - np.argmax(cols[::-1])

    # Tighter inner crop to remove all black corners
    pad = 20
    cropped = img[row_start+pad : row_end-pad,
                  col_start+pad : col_end-pad]

    print(f"✅ Cropped size: {cropped.shape}")
    return cropped


def show_panorama(panorama):
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

    # 1. Load
    img1, img2 = load_images("images/para11.jpg", "images/para12.jpg")

    # 2. Detect
    kp1, des1 = detect_keypoints(img1)
    kp2, des2 = detect_keypoints(img2)
    print(f"\n🔑 Keypoints para11.jpg : {len(kp1)}")
    print(f"🔑 Keypoints para12.jpg : {len(kp2)}")

    # 3. Visualize keypoints
    show_keypoints(img1, kp1, img2, kp2)

    # 4. Match
    good_matches = match_features(des1, des2)

    # 5. Visualize matches
    show_matches(img1, kp1, img2, kp2, good_matches)

    # 6. Homography
    H = compute_homography(kp1, kp2, good_matches)

    # 7. Warp + Gradient Blend
    panorama = warp_and_stitch(img1, img2, H)

    # 8. Crop black borders
    panorama = crop_black_borders(panorama)

    # 9. Final result
    show_panorama(panorama)