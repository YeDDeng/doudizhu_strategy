from PIL import Image
import os
import glob

# 获取第一个文件
files = sorted(glob.glob('dataset/images/database/*.png'))[:1]
if not files:
    files = sorted(glob.glob('dataset/images/database/*.jpg'))[:1]

if files:
    fname = files[0]
    print(f'File: {fname}')

    img = Image.open(fname)
    w, h = img.size
    print(f'Image size: {w}x{h}')
    print(f'Hand card area should be y > {h*0.6:.0f} (60%+)')

    # 读取标签
    label_file = 'dataset/labels/database/' + os.path.basename(fname).replace('.png', '.txt').replace('.jpg', '.txt')

    print(f'Label file: {label_file}')
    print(f'Label exists: {os.path.exists(label_file)}')

    with open(label_file) as f:
        lines = f.readlines()
    print(f'Total labels: {len(lines)}')

    CLASS_NAMES = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2', 'Joker_B', 'Joker_R']

    # 统计不同区域的标签
    top_labels = []   # y < 0.4
    mid_labels = []   # 0.4 <= y < 0.6
    bottom_labels = [] # y >= 0.6

    for line in lines:
        parts = line.strip().split()
        cls, cx, cy, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)
        info = {'cls': cls, 'name': name, 'cx': cx, 'cy': cy, 'bw': bw, 'bh': bh}

        if cy < 0.4:
            top_labels.append(info)
        elif cy < 0.6:
            mid_labels.append(info)
        else:
            bottom_labels.append(info)

    print(f'\nTop area (y < 40%): {len(top_labels)} labels')
    print(f'Mid area (40-60%): {len(mid_labels)} labels')
    print(f'Bottom area (y >= 60%): {len(bottom_labels)} labels')

    print('\nBottom area detected cards:')
    for i, info in enumerate(bottom_labels[:20]):
        x = int(info['cx'] * w)
        y = int(info['cy'] * h)
        bw = int(info['bw'] * w)
        bh = int(info['bh'] * h)
        print(f'  [{i}] {info["name"]}: center=({x},{y}), box={bw}x{bh}, y_ratio={info["cy"]:.3f}')

    print('\nTop area (possible UI):')
    for i, info in enumerate(top_labels[:10]):
        x = int(info['cx'] * w)
        y = int(info['cy'] * h)
        print(f'  [{i}] {info["name"]}: center=({x},{y}), y_ratio={info["cy"]:.3f}')