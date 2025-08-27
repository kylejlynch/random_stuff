"""
Setup script for fraud clustering environment on work computers with package constraints.
This script ensures all compatibility imports are working correctly.
"""

import sys

def setup_sklearn_compatibility():
    """Set up sklearn compatibility imports"""
    try:
        import six
        import joblib
        
        # Fix sklearn.externals compatibility
        sys.modules['sklearn.externals.joblib'] = joblib
        sys.modules['sklearn.externals.six'] = six
        
        print("✓ sklearn compatibility imports configured")
        return True
    except ImportError as e:
        print(f"✗ Error setting up sklearn compatibility: {e}")
        return False

def test_imports():
    """Test all required package imports"""
    packages = [
        ('pandas', '1.1.5'),
        ('numpy', '1.19.5'),
        ('scipy', '1.5.4'),
        ('sklearn', '0.24.2'),
        ('hdbscan', '0.8.20'),
        ('umap', '0.5.3'),
        ('plotly', None),
        ('matplotlib', None),
        ('seaborn', None),
        ('pynndescent', '0.5.13'),
        ('joblib', None),
        ('numba', '0.53.1'),
        ('llvmlite', '0.36.0')
    ]
    
    results = []
    
    for package, expected_version in packages:
        try:
            if package == 'sklearn':
                import sklearn
                module = sklearn
                package_name = 'scikit-learn'
            elif package == 'umap':
                import umap
                module = umap
                package_name = 'umap-learn'
            else:
                module = __import__(package)
                package_name = package
            
            version = getattr(module, '__version__', 'unknown')
            
            if expected_version and version != expected_version:
                status = f"⚠ {package_name}: {version} (expected {expected_version})"
            else:
                status = f"✓ {package_name}: {version}"
            
            results.append((True, status))
            
        except ImportError as e:
            results.append((False, f"✗ {package_name}: Import failed - {e}"))
    
    return results

def main():
    """Main setup and test function"""
    print("Setting up fraud clustering environment for work computer...")
    print("=" * 60)
    
    # Set up compatibility
    sklearn_ok = setup_sklearn_compatibility()
    
    print("\nTesting package imports...")
    print("-" * 30)
    
    # Test imports
    results = test_imports()
    all_ok = True
    
    for success, message in results:
        print(message)
        if not success:
            all_ok = False
    
    print("\n" + "=" * 60)
    
    if all_ok and sklearn_ok:
        print("✓ Environment setup complete! All packages are available.")
        print("\nYou can now run:")
        print("  python example_dataframe_usage.py")
        print("  python fraud_clustering_pipeline.py")
    else:
        print("✗ Some packages are missing or incompatible.")
        print("\nPlease install missing packages using:")
        print("  pip install -r requirements.txt")
        print("  or")
        print("  conda env create -f environment.yml")
    
    return all_ok and sklearn_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)