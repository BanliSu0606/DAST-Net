import os
import glob
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from collections import defaultdict
import argparse

def parse_filename(filename):
    basename = os.path.splitext(os.path.basename(filename))[0]
    parts = basename.split('_')
    performer = int(parts[0][1:])
    action = int(parts[1][1:])
    replication = int(parts[2][1:])
    return performer, action, replication


def load_kinect_data(raw_dir):
    npy_files = sorted(glob.glob(os.path.join(raw_dir, 'P*_A*_R*.npy')))
    
    if len(npy_files) == 0:
        raise ValueError(f"No .npy files found in {raw_dir}!")
    
    data = []
    labels = []
    performers = []
    replications = []
    filenames = []
    frames_per_sequence = []
    
    for f in npy_files:
        try:
            performer, action, replication = parse_filename(f)
            
            skeleton_data = np.load(f).astype(np.float32)
            
            if skeleton_data.ndim != 2:
                print(f"Warning: {os.path.basename(f)} has abnormal dimensions, skipping...")
                continue
                
            if skeleton_data.shape[1] not in [75, 150]:
                print(f"Warning: {os.path.basename(f)} has {skeleton_data.shape[1]} dims, expected 75 or 150, skipping...")
                continue
            
            if skeleton_data.shape[1] == 150:
                skeleton_data = skeleton_data[:, :75]
            
            data.append(skeleton_data)
            labels.append(action)
            performers.append(performer)
            replications.append(replication)
            filenames.append(os.path.basename(f))
            frames_per_sequence.append(skeleton_data.shape[0])
            
        except Exception as e:
            print(f"Error loading {f}: {e}")
            continue
    
    print(f"\nSuccessfully loaded {len(data)} sequences")
    print(f"Total frames: {sum(frames_per_sequence)}")
    print(f"Frames per sequence: min={min(frames_per_sequence)}, max={max(frames_per_sequence)}, mean={np.mean(frames_per_sequence):.1f}")
    
    return data, labels, performers, replications, filenames


def split_dataset_by_performer(data, labels, performers, test_performers=None, test_size=0.2):
    if test_performers is not None:
        train_indices = []
        test_indices = []
        
        for i, p in enumerate(performers):
            if p in test_performers:
                test_indices.append(i)
            else:
                train_indices.append(i)
                
        print(f"\nUsing performers {test_performers} as test set")
        
    else:
        unique_performers = list(set(performers))
        test_performers = np.random.choice(unique_performers, 
                                         size=int(len(unique_performers) * test_size), 
                                         replace=False)
        
        train_indices = []
        test_indices = []
        
        for i, p in enumerate(performers):
            if p in test_performers:
                test_indices.append(i)
            else:
                train_indices.append(i)
        
        print(f"\nRandomly selected performers {test_performers.tolist()} as test set")
    
    train_data = [data[i] for i in train_indices]
    test_data = [data[i] for i in test_indices]
    train_labels = [labels[i] for i in train_indices]
    test_labels = [labels[i] for i in test_indices]
    
    return train_data, test_data, train_labels, test_labels


def save_npz(data, labels, output_path, additional_data=None):
    save_dict = {
        'data': data,  # list of arrays
        'labels': np.array(labels),
        'num_samples': len(data),
        'num_classes': len(set(labels)),
    }
    
    if additional_data:
        for key, value in additional_data.items():
            save_dict[key] = value
    
    np.savez_compressed(output_path, **save_dict)
    print(f"\nSaved to {output_path}")
    print(f"  - Samples: {len(data)}")
    print(f"  - Classes: {save_dict['num_classes']}")


def main(args):
    print("=" * 60)
    print("  Kinect Data to NPZ Converter")
    print("=" * 60)
    
    print("\n[1/4] Loading Kinect data...")
    raw_dir = args.raw_dir
    data, labels, performers, replications, filenames = load_kinect_data(raw_dir)
    
    print("\n[2/4] Dataset statistics...")
    unique_performers = sorted(set(performers))
    unique_actions = sorted(set(labels))
    
    print(f"\nPerformers: {unique_performers}")
    print(f"Actions: {unique_actions}")
    print(f"Total sequences: {len(data)}")
    
    action_counts = defaultdict(int)
    for label in labels:
        action_counts[label] += 1
    
    print("\nSamples per action:")
    for action in sorted(action_counts.keys()):
        print(f"  Action {action}: {action_counts[action]} sequences")
    
    print("\n[3/4] Splitting dataset...")
    
    if args.test_performers:
        test_performers = [int(x) for x in args.test_performers.split(',')]
        train_data, test_data, train_labels, test_labels = split_dataset_by_performer(
            data, labels, performers, test_performers=test_performers
        )
    else:
        train_data, test_data, train_labels, test_labels = split_dataset_by_performer(
            data, labels, performers, test_size=args.test_size
        )
    
    train_frames = sum([seq.shape[0] for seq in train_data])
    test_frames = sum([seq.shape[0] for seq in test_data])
    
    print(f"\nTraining set: {len(train_data)} sequences, {train_frames} frames")
    print(f"Test set: {len(test_data)} sequences, {test_frames} frames")
    print(f"Train/Test ratio: {train_frames/(train_frames+test_frames):.2f}")
    
    train_action_counts = defaultdict(int)
    test_action_counts = defaultdict(int)
    for label in train_labels:
        train_action_counts[label] += 1
    for label in test_labels:
        test_action_counts[label] += 1
    
    print("\nTraining set action distribution:")
    for action in sorted(train_action_counts.keys()):
        print(f"  Action {action}: {train_action_counts[action]} sequences")
    
    print("\nTest set action distribution:")
    for action in sorted(test_action_counts.keys()):
        print(f"  Action {action}: {test_action_counts[action]} sequences")
    
    print("\n[4/4] Saving to NPZ format...")
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train_data.npz')
    save_npz(train_data, train_labels, train_path, 
             additional_data={'split': 'train', 'frames': train_frames})
    
    test_path = os.path.join(output_dir, 'test_data.npz')
    save_npz(test_data, test_labels, test_path,
             additional_data={'split': 'test', 'frames': test_frames})
    
    if args.save_full:
        full_path = os.path.join(output_dir, 'full_dataset.npz')
        save_npz(data, labels, full_path,
                 additional_data={'performers': performers, 'replications': replications})
    
    print("\n" + "=" * 60)
    print("  Conversion complete!")
    print(f"  Output directory: {output_dir}")
    print("=" * 60)
    
    print("\n[Usage Example]")
    print("  # Load training data")
    print("  train_data = np.load('train_data.npz', allow_pickle=True)")
    print("  X_train = train_data['data']  # List of sequences")
    print("  y_train = train_data['labels']  # Labels")
    print("  X_train[0].shape  # (T, 75) - T is variable")
    print("\n  # For fixed-length input, use padding or truncation")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Kinect skeleton data to NPZ format')
    
    parser.add_argument('--raw_dir', type=str, default='./raw_skeletons',
                        help='Directory containing raw .npy files')
    parser.add_argument('--output_dir', type=str, default='./npz_data',
                        help='Directory to save NPZ files')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data for testing (if test_performers not specified)')
    parser.add_argument('--test_performers', type=str, default=None,
                        help='Comma-separated performer IDs to use as test set (e.g., "1,3,5")')
    parser.add_argument('--save_full', action='store_true',
                        help='Also save full dataset (train+test)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    main(args)