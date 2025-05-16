#!/usr/bin/env python
"""
Install SSL certificates for Python on macOS.
This is a common issue on macOS where Python can't verify SSL certificates.
"""
import os
import sys
import subprocess
import ssl

def main():
    """Fix SSL certificate issues on macOS."""
    print("Checking SSL certificate installation...")
    
    try:
        # Try a simple HTTPS request
        import urllib.request
        response = urllib.request.urlopen("https://www.python.org/")
        print("SSL certificates are working correctly!")
        return 0
    except ssl.SSLCertVerificationError:
        print("SSL certificate verification failed.")
        print("This is a common issue on macOS.")
        
        # Find the current Python installation
        python_path = sys.executable
        print(f"Python executable: {python_path}")
        
        # Determine the certificate installation command
        cmd = f"{python_path} -m pip install --upgrade certifi"
        print(f"\nAttempting to fix by installing certifi package...\n{cmd}\n")
        
        try:
            subprocess.run(cmd, shell=True, check=True)
            print("\nCertifi installed successfully.")
        except subprocess.CalledProcessError:
            print("Error installing certifi.")
        
        # Try running the macOS certificate installation script
        if "darwin" in sys.platform.lower():
            print("\nRunning macOS certificate installation script...")
            try:
                # Find the Install Certificates.command script in the Python installation
                python_dir = os.path.dirname(python_path)
                cert_script = os.path.join(python_dir, "Install Certificates.command")
                
                if os.path.exists(cert_script):
                    print(f"Found certificate script: {cert_script}")
                    subprocess.run(f"bash '{cert_script}'", shell=True, check=True)
                    print("Certificate installation script ran successfully.")
                else:
                    cert_script_alt = "/Applications/Python 3.x/Install Certificates.command"
                    print(f"Certificate script not found at expected location.")
                    print(f"Try running: {cert_script_alt}")
                    print("(Replace '3.x' with your Python version)")
            except Exception as e:
                print(f"Error running certificate script: {e}")
        
        print("\nAdditional troubleshooting:")
        print("1. Go to Finder → Applications → Python 3.x → double-click 'Install Certificates.command'")
        print("2. Or add this code at the start of your scripts:")
        print("   import ssl")
        print("   ssl._create_default_https_context = ssl._create_unverified_context")
        print("\nAlternatively, use the '--skip_presidio' option to avoid this issue:")
        print("   ./process_any_data.py --input ~/Downloads/Medical/ --output phi_results --skip_presidio")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())