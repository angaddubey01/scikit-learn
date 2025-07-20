"""
Simple test script that implements the CountVectorizer's critical methods
to test our fix for the get_feature_names method.
"""

class SimplifiedCountVectorizer:
    """Simplified version for testing the fix"""
    
    def __init__(self, vocabulary=None):
        self.vocabulary = vocabulary
        self.fixed_vocabulary_ = False
        
    def _validate_vocabulary(self):
        """Simple implementation of _validate_vocabulary"""
        vocabulary = self.vocabulary
        if vocabulary is not None:
            # Convert vocabulary to dict
            if not isinstance(vocabulary, dict):
                vocab = {}
                for i, t in enumerate(vocabulary):
                    vocab[t] = i
                vocabulary = vocab
            
            self.fixed_vocabulary_ = True
            self.vocabulary_ = dict(vocabulary)
        else:
            self.fixed_vocabulary_ = False
    
    def _check_vocabulary(self):
        """Check if vocabulary is missing (not fit-ed)"""
        if not hasattr(self, 'vocabulary_'):
            raise ValueError("Vocabulary wasn't fitted")
    
    def transform(self, documents):
        """Transform documents - simplified for testing"""
        if not hasattr(self, 'vocabulary_'):
            self._validate_vocabulary()
        
        self._check_vocabulary()
        # In the real implementation, this would transform the documents
        # but for testing we just return a dummy result
        return "Transformed documents using vocabulary"
    
    # Original method with the bug
    def get_feature_names_original(self):
        """Original buggy implementation"""
        self._check_vocabulary()
        return sorted(self.vocabulary_.keys())
    
    # Fixed method
    def get_feature_names(self):
        """Fixed implementation"""
        if not hasattr(self, 'vocabulary_'):
            self._validate_vocabulary()
            
        self._check_vocabulary()
        return sorted(self.vocabulary_.keys())

# Test case from the issue description
vocabulary = ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']

print("Test 1: Standard SimplifiedCountVectorizer without vocabulary")
vectorizer = SimplifiedCountVectorizer()
print(f"Has vocabulary_ before fit?: {hasattr(vectorizer, 'vocabulary_')}")

# This would raise ValueError (equivalent to NotFittedError in sklearn)
try:
    vectorizer.get_feature_names_original()
    print("Error: Expected ValueError but got no exception")
except Exception as e:
    print(f"Expected error with original method: {type(e).__name__}: {e}")

print("\nTest 2: SimplifiedCountVectorizer with vocabulary - our fix")
vectorizer = SimplifiedCountVectorizer(vocabulary=vocabulary)
print(f"Has vocabulary_ before transform?: {hasattr(vectorizer, 'vocabulary_')}")

# This should raise an error with the original method
try:
    vectorizer.get_feature_names_original()
    print("Error: Expected ValueError but got no exception")
except Exception as e:
    print(f"Expected error with original method: {type(e).__name__}: {e}")

# This should work with our fixed method
try:
    feature_names = vectorizer.get_feature_names()
    print(f"Feature names with fixed method: {feature_names}")
    print(f"Has vocabulary_ after get_feature_names?: {hasattr(vectorizer, 'vocabulary_')}")
except Exception as e:
    print(f"Unexpected error with fixed method: {type(e).__name__}: {e}")

# Now transform some data
print("\nTransform documents:")
result = vectorizer.transform(["some document"])
print(f"Transform result: {result}")
print(f"Has vocabulary_ after transform?: {hasattr(vectorizer, 'vocabulary_')}")

# Both methods should work after transform
try:
    feature_names_original = vectorizer.get_feature_names_original()
    feature_names_fixed = vectorizer.get_feature_names()
    print(f"Original method after transform: {feature_names_original}")
    print(f"Fixed method after transform: {feature_names_fixed}")
except Exception as e:
    print(f"Unexpected error after transform: {type(e).__name__}: {e}")