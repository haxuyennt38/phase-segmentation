import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import ndimage
from skimage import filters, measure, io
from skimage.color import rgb2gray
from matplotlib.colors import ListedColormap


# =========================================================
# STEP 1: LOAD PNG IMAGE
# =========================================================
def load_png_image(image_path="standalone_micro_ct_scan.png", shape=(500, 500)):
    """
    Loads a PNG image from disk and converts it to 8-bit grayscale.
    If RGBA (4 channels) or RGB (3 channels), converts to single-channel grayscale.
    If file is missing, checks fallback search paths before generating synthetic sample.
    """
    search_paths = [image_path, os.path.join("/workspace/knowledge", os.path.basename(image_path))]
    found_path = None
    for p in search_paths:
        if os.path.exists(p):
            found_path = p
            break

    if found_path:
        print(f"[INFO] Loading PNG image from disk: {found_path}")
        img = io.imread(found_path)
        if img.ndim == 3:
            if img.shape[2] == 4:
                img = img[:, :, :3]  # Drop alpha channel
            img = (rgb2gray(img) * 255).astype(np.uint8)
        return img.astype(np.uint8)

    print(f"[WARNING] File '{image_path}' not found. Generating synthetic PNG image for testing...")
    np.random.seed(42)
    img = np.full(shape, 155, dtype=np.float32) + np.random.normal(0, 4, shape)

    def draw_ellipse(shape_tuple, center, axes, angle_deg):
        y, x = np.ogrid[:shape_tuple[0], :shape_tuple[1]]
        rad = np.radians(angle_deg)
        x_c, y_c = center
        x_rot = (x - x_c) * np.cos(rad) + (y - y_c) * np.sin(rad)
        y_rot = -(x - x_c) * np.sin(rad) + (y - y_c) * np.cos(rad)
        return (x_rot ** 2 / axes[0] ** 2 + y_rot ** 2 / axes[1] ** 2) <= 1

    mask_b1 = draw_ellipse(shape, (180, 200), (130, 60), 25)
    mask_b2 = draw_ellipse(shape, (340, 320), (140, 70), -35)
    bundles_mask = mask_b1 | mask_b2
    img[bundles_mask] = 240 + np.random.normal(0, 5, np.sum(bundles_mask))

    for _ in range(30):
        bx, by = np.random.randint(50, 450), np.random.randint(50, 450)
        if bundles_mask[by, bx]:
            mv = draw_ellipse(shape, (bx, by), (3, 3), 0)
            img[mv] = 10 + np.random.normal(0, 2, np.sum(mv))

    m1 = draw_ellipse(shape, (350, 180), (25, 15), 15)
    macro_mask = m1 & (~bundles_mask)
    img[macro_mask] = 5 + np.random.normal(0, 2, np.sum(macro_mask))

    img_blur = ndimage.gaussian_filter(img, sigma=1.2)
    return np.clip(img_blur, 0, 255).astype(np.uint8)


# =========================================================
# STEP 2: 4-PHASE SEGMENTATION ALGORITHM
# =========================================================
def segment_4phases(img, sigma=1.2, area_threshold=80):
    """
    Segments a grayscale Micro-CT PNG image into 4 material phases:
      - Phase 0: Inter-bundle Macro-voids (Crimson Red)
      - Phase 1: Intra-bundle Micro-voids/Lumens (Bright Cyan)
      - Phase 2: Free Resin Matrix (Dark Slate)
      - Phase 3: Flax Yarn Bundles (Vibrant Gold/Amber)
    """
    img_filtered = filters.gaussian(img, sigma=sigma, preserve_range=True).astype(np.uint8)
    thresholds = filters.threshold_multiotsu(img_filtered, classes=3)
    seg_3class = np.digitize(img_filtered, bins=thresholds)

    voids_mask = (seg_3class == 0)
    labeled_voids = measure.label(voids_mask)
    props = measure.regionprops(labeled_voids)

    phase_map = np.zeros_like(seg_3class)
    phase_map[seg_3class == 1] = 2  # Free Resin Matrix
    phase_map[seg_3class == 2] = 3  # Flax Yarn Bundles

    for prop in props:
        region_mask = (labeled_voids == prop.label)
        if prop.area >= area_threshold:
            phase_map[region_mask] = 0  # Macro-voids
        else:
            phase_map[region_mask] = 1  # Micro-voids / Lumens

    return img_filtered, thresholds, phase_map


# =========================================================
# STEP 3: AREA / VOLUME FRACTION (%) CALCULATION
# =========================================================
def calculate_fractions(phase_map):
    """Calculates and prints area/volume fractions (%) for each phase"""
    total_pixels = phase_map.size
    phase_names = {
        0: "Phase 0: Inter-bundle Macro-voids",
        1: "Phase 1: Intra-bundle Micro-voids/Lumens",
        2: "Phase 2: Free Resin Matrix",
        3: "Phase 3: Flax Yarn Bundles"
    }

    print("\n" + "=" * 65)
    print(f"{'PHASE NAME':<42} | {'PIXEL COUNT':<10} | {'FRACTION (%)':<10}")
    print("=" * 65)

    results = {}
    for phase_id, name in phase_names.items():
        count = int(np.sum(phase_map == phase_id))
        pct = (count / total_pixels) * 100.0
        results[phase_id] = pct
        print(f"{name:<42} | {count:<10d} | {pct:>9.2f}%")

    print("=" * 65)
    total_porosity = results[0] + results[1]
    print(f"--> TOTAL POROSITY (Phase 0 + Phase 1): {total_porosity:.2f}%\n")
    return results


# =========================================================
# STEP 4: VISUALIZE & SAVE RESULT WITH HIGH-CONTRAST PALETTE
# =========================================================
def plot_results(img_raw, img_filtered, thresholds, phase_map, output_file="standalone_micro_ct_result_v4.png"):
    """Generates 4-panel analysis plot with high-contrast color scheme"""
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), facecolor='#fafafa')
    ax1, ax2, ax3, ax4 = axes.ravel()

    fig.suptitle("Micro-CT 4-Phase PNG Image Analysis (High-Contrast Palette)",
                 fontsize=15, fontweight='bold', color='#1a1a1a', y=0.98)

    # Panel 1: Original Image
    ax1.imshow(img_raw, cmap='gray')
    ax1.set_title("1. Original PNG Input Image", fontsize=11, fontweight='bold', pad=8)
    ax1.axis('off')

    # Panel 2: Histogram & Multi-Otsu Thresholds
    ax2.set_facecolor('#ffffff')
    ax2.hist(img_filtered.ravel(), bins=256, color='#555555', alpha=0.7, edgecolor='none')
    ax2.axvline(thresholds[0], color='#e41a1c', linestyle='--', linewidth=2, label=f'Threshold 1: {thresholds[0]:.1f}')
    ax2.axvline(thresholds[1], color='#377eb8', linestyle='--', linewidth=2, label=f'Threshold 2: {thresholds[1]:.1f}')
    ax2.set_title("2. Gray Level Histogram & Multi-Otsu Thresholds", fontsize=11, fontweight='bold', pad=8)
    ax2.set_xlabel("Pixel Gray Value", fontsize=10)
    ax2.set_ylabel("Pixel Count", fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)

    # Panel 3: Gaussian Denoised Image
    ax3.imshow(img_filtered, cmap='gray')
    ax3.set_title("3. Gaussian Denoised Image", fontsize=11, fontweight='bold', pad=8)
    ax3.axis('off')

    # Panel 4: Clean, high-contrast palette
    palette = ['#d7191c', '#00c5ff', '#2b3e50', '#f39c12']
    custom_cmap = ListedColormap(palette)

    ax4.imshow(phase_map, cmap=custom_cmap, vmin=0, vmax=3)
    ax4.set_title("4. 4-Phase Segmentation Map (High Contrast)", fontsize=11, fontweight='bold', pad=8)
    ax4.axis('off')

    legend_labels = [
        'Phase 0: Inter-bundle Macro-voids (Red)',
        'Phase 1: Intra-bundle Micro-voids/Lumens (Cyan)',
        'Phase 2: Free Resin Matrix (Dark Slate)',
        'Phase 3: Flax Yarn Bundles (Gold/Amber)'
    ]
    patches = [mpatches.Patch(color=palette[i], label=legend_labels[i]) for i in range(4)]
    ax4.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=9, frameon=True,
               facecolor='#ffffff')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_file, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Output figure saved to '{output_file}'")


# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    PNG_FILE_PATH = "standalone_micro_ct_scan.png"

    # 1. Load PNG image
    img_raw = load_png_image(image_path=PNG_FILE_PATH)

    # 2. Run 4-Phase Segmentation
    img_filtered, thresholds, phase_map = segment_4phases(img_raw, sigma=1.2, area_threshold=80)

    # 3. Calculate Fractions (%)
    calculate_fractions(phase_map)

    # 4. Save Plot
    plot_results(img_raw, img_filtered, thresholds, phase_map, output_file="standalone_micro_ct_result.png")