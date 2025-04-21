"""
Dependency checking and installation utilities.
"""
import sys
import subprocess

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {title} ".center(60, "="))
    print("=" * 60)

def check_dependencies(skip=False):
    """Check and install missing dependencies."""
    if skip:
        return True
        
    print_header("Checking Dependencies")
    
    required_packages = [
        "pyyaml", "ipywidgets", "typing-extensions", "attrs", "ninja"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.lower())
            print(f"✅ {package} installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} missing")
    
    if missing_packages:
        print(f"\nInstalling {len(missing_packages)} missing packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        print("✅ Missing packages installed successfully")
    
    # Check for Shap-E installation
    try:
        import shap_e
        print("✅ Shap-E package installed")
    except ImportError:
        print("Installing Shap-E as a package...")
        import os
        # Get the repository root path
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        shap_e_path = os.path.join(script_dir, 'shap-e')
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", shap_e_path])
        print("✅ Shap-E installed successfully")
    
    return True