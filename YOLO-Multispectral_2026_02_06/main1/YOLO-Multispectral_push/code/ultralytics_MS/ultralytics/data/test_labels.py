from pathlib import Path

label_dir = Path("D:/PD/yolo_mod/data/WeedsGalore/data_processed/train_val_test_V7/RGBNR/test/labels")

for label_file in label_dir.glob("*.txt"):
    content = label_file.read_text().strip()
    if not content:
        print(f"[EMPTY] {label_file}")
    elif any(len(line.split()) != 5 for line in content.splitlines()):
        print(f"[BAD FORMAT] {label_file}")