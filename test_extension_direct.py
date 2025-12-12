#!/usr/bin/env python3
"""Test C++ extension import with detailed error reporting"""

import sys
import os

print("=" * 60)
print("Testing C++ Extension Import")
print("=" * 60)

# Check package location
try:
    import embodik
    print(f"✓ Package imported from: {embodik.__file__}")
    print(f"  Package directory: {os.path.dirname(embodik.__file__)}")
except Exception as e:
    print(f"✗ Failed to import embodik: {e}")
    sys.exit(1)

# Check if extension file exists
package_dir = os.path.dirname(embodik.__file__)
extension_files = [f for f in os.listdir(package_dir) if f.startswith('_embodik_impl') and f.endswith('.so')]
print(f"\nExtension files in package dir: {extension_files}")

# Try to import extension directly
print("\nAttempting direct import...")
try:
    import importlib.util
    extension_path = os.path.join(package_dir, extension_files[0]) if extension_files else None
    if extension_path and os.path.exists(extension_path):
        print(f"  Found extension at: {extension_path}")
        spec = importlib.util.spec_from_file_location("_embodik_impl", extension_path)
        if spec and spec.loader:
            impl = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(impl)
            print("  ✓ Extension loaded directly!")
            print(f"  Available symbols: {len([x for x in dir(impl) if not x.startswith('_')])} public items")
        else:
            print("  ✗ Failed to create spec")
    else:
        print(f"  ✗ Extension file not found at: {extension_path}")
except Exception as e:
    print(f"  ✗ Error loading extension: {e}")
    import traceback
    traceback.print_exc()

# Try normal import
print("\nAttempting normal import...")
try:
    from embodik import _embodik_impl
    print("  ✓ Extension imported via normal import!")
    print(f"  Available symbols: {len([x for x in dir(_embodik_impl) if not x.startswith('_')])} public items")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
