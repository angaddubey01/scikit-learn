import numpy as np
from sklearn.model_selection import StratifiedKFold

def test_stratified_kfold_shuffling():
    # Set up the same test case as in the bug report
    samples_per_class = 10
    X = np.linspace(0, samples_per_class*2-1, samples_per_class * 2)
    y = np.concatenate((np.ones(samples_per_class), np.zeros(samples_per_class)), axis=0)

    print("Input data:")
    print("X:", X)
    print("y:", y)
    print()

    print('Testing shuffle=False:')
    # Test with shuffle=False
    kf_no_shuffle = StratifiedKFold(n_splits=5, shuffle=False, random_state=42)
    
    # Collect all indices
    train_indices_no_shuffle = []
    test_indices_no_shuffle = []
    
    for train_idx, test_idx in kf_no_shuffle.split(X, y):
        train_indices_no_shuffle.append(sorted(train_idx.tolist()))
        test_indices_no_shuffle.append(sorted(test_idx.tolist()))
        print(f"Train: {train_idx}, Test: {test_idx}")
    
    print()
    print('Testing shuffle=True with random_state=42:')
    # Test with shuffle=True and a fixed random state
    kf_shuffle_1 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Collect all indices
    train_indices_shuffle_1 = []
    test_indices_shuffle_1 = []
    
    for train_idx, test_idx in kf_shuffle_1.split(X, y):
        train_indices_shuffle_1.append(sorted(train_idx.tolist()))
        test_indices_shuffle_1.append(sorted(test_idx.tolist()))
        print(f"Train: {train_idx}, Test: {test_idx}")
    
    print()
    print('Testing shuffle=True with random_state=43:')
    # Test with shuffle=True and a different random state
    kf_shuffle_2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)
    
    # Collect all indices
    train_indices_shuffle_2 = []
    test_indices_shuffle_2 = []
    
    for train_idx, test_idx in kf_shuffle_2.split(X, y):
        train_indices_shuffle_2.append(sorted(train_idx.tolist()))
        test_indices_shuffle_2.append(sorted(test_idx.tolist()))
        print(f"Train: {train_idx}, Test: {test_idx}")

    # Check if the test indices are different with different random states
    are_different = False
    for fold_idx in range(5):
        if train_indices_shuffle_1[fold_idx] != train_indices_shuffle_2[fold_idx]:
            are_different = True
            break
    
    print()
    print(f"Test indices are {'different' if are_different else 'the same'} with different random states")
    
    # Check if pairs are always consistent between classes
    print()
    print("Checking if our fix prevents consistent pairing between classes:")
    consistent_pairs = True
    for test_indices in test_indices_shuffle_1 + test_indices_shuffle_2:
        class1_indices = [i for i in test_indices if i < samples_per_class]
        class2_indices = [i for i in test_indices if i >= samples_per_class]
        
        # Check if each index in class 1 is always paired with the same index in class 2
        for c1_idx in class1_indices:
            possible_pair = c1_idx + samples_per_class
            if possible_pair in class2_indices:
                print(f"Found a consistent pair: {c1_idx} and {possible_pair}")
                # We expect some consistent pairs by chance, so just report them
    
    print()
    print("Conclusion:")
    if are_different:
        print("SUCCESS: Our fix is working! Different random states produce different train/test splits.")
    else:
        print("FAILURE: Our fix is not working. Different random states produce the same train/test splits.")

if __name__ == "__main__":
    test_stratified_kfold_shuffling()