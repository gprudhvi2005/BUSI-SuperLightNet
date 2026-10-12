"""
split_dataset.py — Create stratified train/val/test splits for BUSI.

Usage:
    python split_dataset.py --data_dir /path/to/Dataset_BUSI_with_GT

Outputs (in --output_dir):
    train.txt   val.txt   test.txt
Each file contains one *relative* image path per line (relative to data_dir).
"""
import os
import argparse
import random
from collections import Counter
from dataset import find_busi_pairs


def main():
    parser = argparse.ArgumentParser(
        description="Stratified train/val/test split for the BUSI dataset")
    parser.add_argument('--data_dir',    type=str, required=True,
                        help='Root directory containing benign/, malignant/, normal/')
    parser.add_argument('--output_dir',  type=str, default='.',
                        help='Where to write train.txt / val.txt / test.txt')
    parser.add_argument('--train_ratio', type=float, default=0.70)
    parser.add_argument('--val_ratio',   type=float, default=0.15)
    # test_ratio = 1 - train_ratio - val_ratio
    parser.add_argument('--categories',  nargs='+',
                        default=['benign', 'malignant', 'normal'])
    parser.add_argument('--seed',        type=int, default=2024)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    all_pairs = find_busi_pairs(args.data_dir, categories=args.categories)
    print(f"\nTotal images found: {len(all_pairs)}")
    counts = Counter(cat for _, _, cat in all_pairs)
    for cat, n in sorted(counts.items()):
        print(f"  {cat}: {n}")

    train_lines, val_lines, test_lines = [], [], []

    for cat in args.categories:
        cat_pairs = [(img, masks) for img, masks, c in all_pairs if c == cat]
        random.shuffle(cat_pairs)
        n       = len(cat_pairs)
        n_train = int(n * args.train_ratio)
        n_val   = int(n * args.val_ratio)

        for img_path, _ in cat_pairs[:n_train]:
            train_lines.append(os.path.relpath(img_path, args.data_dir))
        for img_path, _ in cat_pairs[n_train:n_train + n_val]:
            val_lines.append(os.path.relpath(img_path, args.data_dir))
        for img_path, _ in cat_pairs[n_train + n_val:]:
            test_lines.append(os.path.relpath(img_path, args.data_dir))

    def _write(fname, lines):
        path = os.path.join(args.output_dir, fname)
        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f"  {fname}: {len(lines)} samples  ->  {path}")

    print("\nSplit (stratified per category):")
    _write('train.txt', train_lines)
    _write('val.txt',   val_lines)
    _write('test.txt',  test_lines)
    print(f"\nDone. Train={len(train_lines)} | Val={len(val_lines)} | Test={len(test_lines)}")


if __name__ == '__main__':
    main()

