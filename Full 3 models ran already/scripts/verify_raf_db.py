import os, json

splits = {
    "train": ("data/raf_db/train", 11046),
    "val":   ("data/raf_db/val",    1225),
    "test":  ("data/raf_db/test",   3068)
}

with open("data/label_map.json") as f:
    label_map = json.load(f)

print("=== RAF-DB Verification ===\n")
total = 0
for split, (path, expected) in splits.items():
    if not os.path.exists(path):
        print(f"[{split}] MISSING folder: {path}")
        continue
    split_count = 0
    print(f"[{split}]")
    for cls_idx in range(7):
        cls_folder = os.path.join(path, str(cls_idx))
        if os.path.exists(cls_folder):
            imgs = [f for f in os.listdir(cls_folder)
                    if f.lower().endswith((".png",".jpg",".jpeg"))]
            emotion = label_map["idx_to_emotion"][str(cls_idx)]
            print(f"  [{cls_idx}] {emotion}: {len(imgs)} images")
            split_count += len(imgs)
        else:
            print(f"  [{cls_idx}] NOT FOUND")
    status = "✅" if split_count == expected else f"⚠️  expected {expected}"
    print(f"  Subtotal: {split_count} {status}\n")
    total += split_count
print(f"Total RAF-DB images: {total}")
