"""
Helper script to set up Kaggle credentials correctly.
This ensures the kaggle.json file is in the right location
and datasets will be downloaded to the proper folder.
"""

import os
import json
import shutil
import stat
import sys

def setup_kaggle_credentials():
    # Define paths
    home_dir = os.path.expanduser('~')
    kaggle_dir = os.path.join(home_dir, '.kaggle')
    kaggle_json_path = os.path.join(kaggle_dir, 'kaggle.json')
    downloads_kaggle_json = os.path.join(home_dir, 'Downloads', 'kaggle.json')
    
    # Create .kaggle directory if it doesn't exist
    if not os.path.exists(kaggle_dir):
        print(f"Creating .kaggle directory at {kaggle_dir}")
        os.makedirs(kaggle_dir, exist_ok=True)
    
    # Check if kaggle.json already exists in the right place
    if os.path.exists(kaggle_json_path):
        print(f"Kaggle credentials already exist at {kaggle_json_path}")
        
        # Verify the contents are correct
        try:
            with open(kaggle_json_path, 'r') as f:
                creds = json.load(f)
                if creds.get('username') == 'shanszu':
                    print("Credentials appear to be correctly configured.")
                else:
                    print("Credentials exist but username doesn't match expected value.")
                    should_replace = input("Would you like to replace with correct credentials? (y/n): ")
                    if should_replace.lower() == 'y':
                        create_credentials_file(kaggle_json_path)
        except Exception as e:
            print(f"Error reading existing credentials: {e}")
            should_replace = input("Would you like to replace with correct credentials? (y/n): ")
            if should_replace.lower() == 'y':
                create_credentials_file(kaggle_json_path)
    
    # If kaggle.json doesn't exist in .kaggle directory but exists in Downloads, copy it
    elif os.path.exists(downloads_kaggle_json):
        print(f"Found kaggle.json in Downloads folder, copying to {kaggle_json_path}")
        shutil.copy2(downloads_kaggle_json, kaggle_json_path)
        
        # Set proper permissions
        try:
            if os.name == 'nt':  # Windows
                os.chmod(kaggle_json_path, stat.S_IREAD)
            else:  # Linux/Mac
                os.chmod(kaggle_json_path, 0o600)
            print("Set proper permissions on kaggle.json")
        except Exception as e:
            print(f"Warning: Couldn't set permissions on kaggle.json: {e}")
    
    # Otherwise, create the credentials file from scratch
    else:
        print(f"No existing kaggle.json found, creating at {kaggle_json_path}")
        create_credentials_file(kaggle_json_path)

def create_credentials_file(file_path):
    """Create a new kaggle.json file with the correct credentials"""
    credentials = {
        "username": "shanszu",
        "key": "d85de9733fa04f82a3688f73af58485a"
    }
    
    try:
        with open(file_path, 'w') as f:
            json.dump(credentials, f)
        
        # Set proper permissions
        if os.name == 'nt':  # Windows
            os.chmod(file_path, stat.S_IREAD)
        else:  # Linux/Mac
            os.chmod(file_path, 0o600)
        
        print(f"Successfully created kaggle.json at {file_path}")
    except Exception as e:
        print(f"Error creating kaggle.json: {e}")

def verify_kaggle_installation():
    """Check if kaggle package is installed, install if not"""
    try:
        import kaggle
        print("Kaggle package is installed")
        return True
    except ImportError:
        print("Kaggle package is not installed")
        install = input("Would you like to install the kaggle package? (y/n): ")
        if install.lower() == 'y':
            import subprocess
            print("Installing kaggle package...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
            print("Kaggle package installed successfully")
            return True
        return False

def main():
    print("Genesis-Sphere Kaggle Setup Helper")
    print("=================================")
    
    # Ensure the kaggle.json file is in the right place
    setup_kaggle_credentials()
    
    # Verify kaggle package is installed
    if verify_kaggle_installation():
        print("\nKaggle is properly configured!")
        print("You can now run the download_datasets.py script to download datasets")
        print("All datasets will be saved to the validation/datasets directory")
    else:
        print("\nPlease install the kaggle package with:")
        print("pip install kaggle")
        print("Then run this script again")
    
    print("\nFor troubleshooting, visit: https://github.com/Kaggle/kaggle-api")

if __name__ == "__main__":
    main()
