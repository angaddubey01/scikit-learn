import sys
import os

# Add the current directory to the path so Python can find the module
sys.path.insert(0, os.path.abspath('.'))

# Import the modified CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Test case from the issue description
corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]

vocabulary = ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']

print("Test 1: Standard CountVectorizer without vocabulary")
vectorizer = CountVectorizer()
print(f"Has vocabulary_ before fit?: {hasattr(vectorizer, 'vocabulary_')}")

# This would raise NotFittedError before our fix
try:
    vectorizer.get_feature_names()
    print("Error: Expected NotFittedError but got no exception")
except Exception as e:
    print(f"Expected error: {type(e).__name__}: {e}")

print("\nTest 2: CountVectorizer with vocabulary - our fix")
vectorizer = CountVectorizer(vocabulary=vocabulary)
print(f"Has vocabulary_ before transform?: {hasattr(vectorizer, 'vocabulary_')}")

# This should work now with our fix
try:
    feature_names = vectorizer.get_feature_names()
    print(f"Feature names without transform: {feature_names}")
    print(f"Has vocabulary_ after get_feature_names?: {hasattr(vectorizer, 'vocabulary_')}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")

# Now transform some data
print("\nTransform corpus:")
X = vectorizer.transform(corpus)
print(f"Transform output shape: {X.shape}")
print(f"Has vocabulary_ after transform?: {hasattr(vectorizer, 'vocabulary_')}")

# And get_feature_names should still work
try:
    feature_names = vectorizer.get_feature_names()
    print(f"Feature names after transform: {feature_names}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")