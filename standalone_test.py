import numpy as np
from collections import defaultdict

class FixedStratifiedKFold:
    """Implementation of our fixed StratifiedKFold class for testing purposes."""
    
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X, y):
        """Generate indices to split data into training and test sets."""
        n_samples = len(y)
        unique_y, y_inversed = np.unique(y, return_inverse=True)
        y_counts = np.bincount(y_inversed)
        
        # Calculate test fold sizes for each class
        fold_sizes = np.full(self.n_splits, y_counts[0] // self.n_splits, dtype=int)
        fold_sizes[:y_counts[0] % self.n_splits] += 1
        
        # Initialize test_folds array
        test_folds = np.zeros(n_samples, dtype=int)
        
        # Get indices for each class
        class_indices = {}
        for cls in unique_y:
            cls_indices = np.where(y == cls)[0]
            if self.shuffle:
                rng = np.random.RandomState(self.random_state)
                cls_indices = rng.permutation(cls_indices)
            class_indices[cls] = cls_indices
        
        # Assign samples to folds
        for cls in unique_y:
            cls_indices = class_indices[cls]
            start = 0
            for fold_idx in range(self.n_splits):
                fold_size = fold_sizes[fold_idx]
                end = start + fold_size
                if end > len(cls_indices):
                    end = len(cls_indices)
                test_folds[cls_indices[start:end]] = fold_idx
                start = end
        
        # Generate train and test indices for each fold
        for i in range(self.n_splits):
            test_idx = np.where(test_folds == i)[0]
            train_idx = np.where(test_folds != i)[0]
            yield train_idx, test_idx


def test_fixed_stratified_kfold():
    # Set up a test case similar to the bug report
    samples_per_class = 10
    X = np.linspace(0, samples_per_class*2-1, samples_per_class * 2)
    y = np.concatenate((np.ones(samples_per_class), np.zeros(samples_per_class)), axis=0)
    
    print("Input data:")
    print("X:", X)
    print("y:", y)
    print()
    
    # Test with shuffle=False
    print("Testing shuffle=False:")
    skf_no_shuffle = FixedStratifiedKFold(n_splits=5, shuffle=False, random_state=42)
    for train_idx, test_idx in skf_no_shuffle.split(X, y):
        print(f"Train: {train_idx}, Test: {test_idx}")
    
    print()
    
    # Test with shuffle=True and random_state=42
    print("Testing shuffle=True with random_state=42:")
    skf_shuffle_1 = FixedStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results_1 = []
    for train_idx, test_idx in skf_shuffle_1.split(X, y):
        print(f"Train: {train_idx}, Test: {test_idx}")
        results_1.append((train_idx.copy(), test_idx.copy()))
    
    print()
    
    # Test with shuffle=True and random_state=43
    print("Testing shuffle=True with random_state=43:")
    skf_shuffle_2 = FixedStratifiedKFold(n_splits=5, shuffle=True, random_state=43)
    results_2 = []
    for train_idx, test_idx in skf_shuffle_2.split(X, y):
        print(f"Train: {train_idx}, Test: {test_idx}")
        results_2.append((train_idx.copy(), test_idx.copy()))
    
    print()
    
    # Check if the splits are different with different random states
    all_same = True
    for i in range(len(results_1)):
        train1, test1 = results_1[i]
        train2, test2 = results_2[i]
        if not np.array_equal(train1, train2) or not np.array_equal(test1, test2):
            all_same = False
            break
    
    print(f"Are all splits the same with different random states? {'Yes' if all_same else 'No'}")
    print(f"Our fix {'is NOT' if all_same else 'is'} working correctly.")
    
    # Check for consistent pairs between classes
    print("\nChecking if indices are consistently paired between classes:")
    pairs = defaultdict(int)
    for _, test_idx in results_1 + results_2:
        class1_indices = [idx for idx in test_idx if idx < samples_per_class]
        class2_indices = [idx for idx in test_idx if idx >= samples_per_class]
        
        for i in class1_indices:
            for j in class2_indices:
                pairs[(i, j)] += 1
    
    # Find pairs that appear in multiple folds
    consistent_pairs = [(pair, count) for pair, count in pairs.items() if count > 1]
    if consistent_pairs:
        print(f"Found {len(consistent_pairs)} consistent pairs out of {len(pairs)} total pairs.")
        # We expect some pairs by chance, but not all
        if len(consistent_pairs) < len(pairs) / 2:
            print("This is likely by chance and NOT due to a problem.")
        else:
            print("This is too many consistent pairs - our fix might not be working correctly.")
    else:
        print("No consistent pairs found - our fix is working correctly.")


if __name__ == "__main__":
    test_fixed_stratified_kfold()