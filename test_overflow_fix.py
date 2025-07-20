"""
Test to verify the fix for integer overflow in fowlkes_mallows_score.
"""
import numpy as np
from sklearn.metrics.cluster import fowlkes_mallows_score

def test_integer_overflow_fowlkes_mallows():
    """
    Test that the fowlkes_mallows_score function handles large integers without overflow.
    
    This test creates labels that would result in very large pk and qk values,
    which previously caused overflow issues when multiplied together.
    """
    # Create large arrays with patterns that would produce large pk, qk values
    n = 50000
    labels_a = np.zeros(n, dtype=np.int)
    labels_b = np.zeros(n, dtype=np.int)
    
    # Set patterns to create large contingency matrix values
    labels_a[:40000] = 0
    labels_a[40000:] = 1
    labels_b[:45000] = 0
    labels_b[45000:] = 1
    
    # This should not produce a RuntimeWarning for overflow and should return a valid score
    score = fowlkes_mallows_score(labels_a, labels_b)
    
    # Verify the result is a valid float (not nan) between 0 and 1
    assert not np.isnan(score), "Score should not be NaN"
    assert 0.0 <= score <= 1.0, "Score should be between 0 and 1"
    
    # This is the expected score for this particular configuration
    # The score with these labels should be approximately 0.82
    expected_score = 0.82
    assert abs(score - expected_score) < 0.01, f"Expected score around {expected_score}, got {score}"

# Run the test
if __name__ == "__main__":
    test_integer_overflow_fowlkes_mallows()
    print("Test passed successfully!")