"""
setup_env.py - Check and install dependencies
Run this script in your conda environment: python setup_env.py
"""

import subprocess
import sys

def run_pip(args):
    """Run pip command"""
    cmd = [sys.executable, "-m", "pip"] + args
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0

def check_import(module_name):
    """Check if module can be imported"""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

def main():
    print("=" * 50)
    print("Dependency Setup for Drone Path Planning")
    print("=" * 50)
    print(f"Python: {sys.executable}")
    print(f"Version: {sys.version}")
    print()

    # Step 1: Install numpy first (required by airsim)
    print("[1/6] Installing numpy...")
    if not check_import("numpy"):
        run_pip(["install", "numpy"])
    else:
        print("  numpy already installed")

    # Step 2: Install msgpack-rpc-python (required by airsim)
    print("\n[2/6] Installing msgpack-rpc-python...")
    run_pip(["install", "msgpack-rpc-python"])

    # Step 3: Install airsim
    print("\n[3/6] Installing airsim...")
    if not check_import("airsim"):
        # Try installing with --no-build-isolation to use existing numpy
        success = run_pip(["install", "--no-build-isolation", "airsim"])
        if not success:
            print("  Trying alternative method...")
            run_pip(["install", "airsim"])
    else:
        print("  airsim already installed")

    # Step 4: Install opencv
    print("\n[4/6] Installing opencv-python...")
    if not check_import("cv2"):
        run_pip(["install", "opencv-python"])
    else:
        print("  opencv-python already installed")

    # Step 5: Install matplotlib and pandas
    print("\n[5/6] Installing matplotlib and pandas...")
    run_pip(["install", "matplotlib", "pandas"])

    # Step 6: Install scipy
    print("\n[6/6] Installing scipy...")
    if not check_import("scipy"):
        run_pip(["install", "scipy"])
    else:
        print("  scipy already installed")

    # Verify installation
    print("\n" + "=" * 50)
    print("Verification")
    print("=" * 50)

    packages = ["numpy", "airsim", "cv2", "matplotlib", "pandas", "scipy"]
    all_ok = True

    for pkg in packages:
        if check_import(pkg):
            print(f"  [OK] {pkg}")
        else:
            print(f"  [FAIL] {pkg}")
            all_ok = False

    print()
    if all_ok:
        print("All dependencies installed successfully!")
        print("You can now run: python fly_planned_path.py")
    else:
        print("Some packages failed to install.")
        print("Please try manual installation.")

if __name__ == "__main__":
    main()
