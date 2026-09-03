"""Quick test for ContourRegistry save/load."""
from src.grain_detection.contour_registry import ContourRegistry

params = {
    "model_type": "u2netp",
    "saliency_threshold": 0.54,
    "adaptive_k": 0.33,
    "min_area": 10002,
    "max_area": 500000,
    "morph_kernel_size": 3,
    "min_circularity": 0.48,
    "crop_padding": 0.2,
    "crop_size": 256,
}

for split in ["train", "val", "test"]:
    root = f"data/processed/{split}"
    p = ContourRegistry.save(root, params)
    print(f"Saved: {p}")

    r = ContourRegistry.load(root)
    s = r["summary"]
    print(
        f"  {split}: {s['total_images']} imgs, "
        f"{s['images_with_seg']} with seg, "
        f"{s['total_grains']} grains, "
        f"{s['num_classes']} classes"
    )
    print(f"  Stale: {ContourRegistry.is_stale(root)}")
    print(f"  Params saved: {list(r['detection_params'].keys())}")
    print()

print("All registries saved and loaded successfully.")
