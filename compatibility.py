"""
Compatibility module for sklearn and other package compatibility issues
This module handles compatibility issues with older package versions on constrained work environments.
"""

# sklearn.externals compatibility for older scikit-learn versions
import sys
import six
import joblib

# Fix for sklearn.externals deprecation in newer versions
sys.modules['sklearn.externals.joblib'] = joblib
sys.modules['sklearn.externals.six'] = six

# Import this at the top of any module that uses sklearn
def ensure_sklearn_compatibility():
    """
    Ensure sklearn compatibility imports are loaded
    Call this function early in your script if needed
    """
    pass  # The imports above are sufficient