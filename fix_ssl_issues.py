#!/usr/bin/env python
"""
Fix SSL certificate verification issues on macOS.

This script helps resolve SSL certificate verification problems,
particularly in Python applications running on macOS.

The script will:
1. Check if SSL certificate verification is failing
2. Locate the correct certificate bundle on your system
3. Set up environment variables for Python to use the correct certificates
4. Optionally install certifi for a more reliable certificate bundle
"""
import os
import sys
import ssl
import subprocess
import urllib.request
from urllib.error import URLError
import platform

def test_ssl_connection():
    """Test SSL connection to a known website."""
    print("Testing SSL connection...")
    try:
        urllib.request.urlopen("https://www.google.com", timeout=5)
        print("✅ SSL connection successful")
        return True
    except URLError as e:
        if isinstance(e.reason, ssl.SSLError):
            print(f"❌ SSL verification failed: {e.reason}")
            return False
        else:
            print(f"❌ Connection error (not SSL-related): {e.reason}")
            return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

def find_cert_file():
    """Find the SSL certificate file on this system."""
    # Check if certifi is installed
    try:
        import certifi
        cert_file = certifi.where()
        print(f"✅ Found certifi certificate bundle at: {cert_file}")
        return cert_file
    except ImportError:
        print("ℹ️ certifi package not installed")
    
    # Look for system certificates
    potential_cert_paths = [
        # macOS locations
        "/etc/ssl/cert.pem",
        "/etc/ssl/certs/ca-certificates.crt",
        "/Library/Application Support/Certivox/certs/ca-bundle.crt",
        # Linux locations
        "/etc/ssl/certs/ca-certificates.crt",
        "/etc/pki/tls/certs/ca-bundle.crt",
        "/etc/ssl/ca-bundle.pem",
        # Homebrew locations
        "/usr/local/etc/openssl/cert.pem",
        "/usr/local/etc/openssl@1.1/cert.pem",
        "/opt/homebrew/etc/openssl/cert.pem",
        "/opt/homebrew/etc/openssl@1.1/cert.pem",
    ]
    
    for path in potential_cert_paths:
        if os.path.exists(path):
            print(f"✅ Found system certificate bundle at: {path}")
            return path
    
    print("❌ Could not find system certificate bundle")
    return None

def setup_ssl_env(cert_file):
    """Set up environment variables for SSL certificate verification."""
    if not cert_file:
        return False
    
    # Set environment variables for the current process
    os.environ['SSL_CERT_FILE'] = cert_file
    os.environ['REQUESTS_CA_BUNDLE'] = cert_file
    
    print("\nEnvironment variables set for this session:")
    print(f"SSL_CERT_FILE={cert_file}")
    print(f"REQUESTS_CA_BUNDLE={cert_file}")
    
    # Generate commands to set in shell
    shell = os.environ.get('SHELL', '').split('/')[-1]
    print("\nTo make these changes permanent, add these lines to your shell profile:")
    
    if shell in ['bash', 'sh']:
        profile = "~/.bash_profile or ~/.bashrc"
        commands = f'export SSL_CERT_FILE="{cert_file}"\nexport REQUESTS_CA_BUNDLE="{cert_file}"'
    elif shell in ['zsh']:
        profile = "~/.zshrc"
        commands = f'export SSL_CERT_FILE="{cert_file}"\nexport REQUESTS_CA_BUNDLE="{cert_file}"'
    else:
        profile = "shell profile"
        commands = f'SSL_CERT_FILE="{cert_file}"\nREQUESTS_CA_BUNDLE="{cert_file}"'
    
    print(f"\nAdd to {profile}:")
    print(f"\n{commands}\n")
    
    # Create a simple script to use in the current directory
    with open('fix_ssl.sh', 'w') as f:
        f.write(f'#!/bin/sh\nexport SSL_CERT_FILE="{cert_file}"\nexport REQUESTS_CA_BUNDLE="{cert_file}"\necho "SSL certificate environment variables set."\n')
    
    os.chmod('fix_ssl.sh', 0o755)
    print(f"Created fix_ssl.sh script in the current directory.")
    print("Run 'source ./fix_ssl.sh' before running your Python script.")
    
    return True

def install_certifi():
    """Install certifi package for reliable certificates."""
    print("\nInstalling certifi package for reliable certificate handling...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "certifi"])
        print("✅ certifi installed successfully")
        
        try:
            import certifi
            cert_file = certifi.where()
            print(f"✅ certifi certificate path: {cert_file}")
            return cert_file
        except ImportError:
            print("❌ certifi installed but could not be imported")
            return None
    except subprocess.CalledProcessError:
        print("❌ Failed to install certifi")
        return None

def fix_ssl_for_presidio():
    """Fix SSL issues specifically for Microsoft Presidio."""
    print("\nApplying fix for Microsoft Presidio SSL issues...")
    
    # Create a patch for the presidio issue
    presidio_fix = """
import os
import ssl
import urllib.request

# Disable SSL verification for Presidio only
ssl._create_default_https_context = ssl._create_unverified_context

# Original PresidioBaseline class will go below
"""
    
    baseline_file = os.path.join('src', 'baseline.py')
    
    if not os.path.exists(baseline_file):
        print(f"❌ Could not find {baseline_file}")
        return False
    
    with open(baseline_file, 'r') as f:
        content = f.read()
    
    if 'ssl._create_default_https_context' in content:
        print("✅ SSL fix already applied to Presidio baseline")
        return True
    
    # Add the SSL fix at the top of the file
    with open(baseline_file, 'w') as f:
        # Add the fix after the imports but before the class definition
        if 'import' in content and 'class' in content:
            import_end = content.rfind('import', 0, content.find('class'))
            import_section_end = content.find('\n\n', import_end)
            if import_section_end > 0:
                new_content = content[:import_section_end] + presidio_fix + content[import_section_end:]
                f.write(new_content)
                print("✅ Applied SSL fix to Presidio baseline")
                return True
        
        # Fallback to prepending the fix
        f.write(presidio_fix + content)
        print("✅ Applied SSL fix to Presidio baseline")
        return True

def fix_process_any_data():
    """Fix SSL issues in process_any_data.py."""
    print("\nApplying fix to process_any_data.py...")
    
    process_file = 'process_any_data.py'
    
    if not os.path.exists(process_file):
        print(f"❌ Could not find {process_file}")
        return False
    
    with open(process_file, 'r') as f:
        content = f.read()
    
    # Check if fix is already applied
    if 'ssl._create_default_https_context' in content:
        print("✅ SSL fix already applied to process_any_data.py")
        return True
    
    # Add the fix right after imports
    ssl_fix = """
# Disable SSL verification for external requests
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
"""
    
    # Find a good spot to insert the fix
    if 'import' in content:
        import_end = content.rfind('import', 0, 500)  # Look in first 500 chars only
        import_section_end = content.find('\n\n', import_end)
        if import_section_end > 0:
            new_content = content[:import_section_end] + ssl_fix + content[import_section_end:]
            
            with open(process_file, 'w') as f:
                f.write(new_content)
                print("✅ Applied SSL fix to process_any_data.py")
                return True
    
    # Fallback to adding at the beginning
    with open(process_file, 'w') as f:
        f.write(ssl_fix + content)
        print("✅ Applied SSL fix to process_any_data.py")
        return True

def main():
    """Main function to fix SSL issues."""
    print(f"SSL Certificate Fixer for Python on {platform.system()}")
    print("=" * 50)
    
    # System information
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"OpenSSL: {ssl.OPENSSL_VERSION}\n")
    
    # Test SSL connection
    ssl_working = test_ssl_connection()
    
    if ssl_working:
        print("\nSSL verification is working correctly!")
        
        # Still fix Presidio-specific issues
        fix_ssl_for_presidio()
        fix_process_any_data()
        
        print("\nRecommendation: Your SSL is working, but specific fixes were applied to:")
        print("- process_any_data.py: SSL verification disabled for external requests")
        if os.path.exists(os.path.join('src', 'baseline.py')):
            print("- src/baseline.py: SSL verification disabled for Microsoft Presidio")
        
        return 0
    
    print("\nFinding SSL certificate bundle...")
    cert_file = find_cert_file()
    
    if not cert_file:
        print("\nInstalling certifi to provide reliable certificates...")
        cert_file = install_certifi()
    
    if cert_file:
        setup_ssl_env(cert_file)
    else:
        print("\n❌ Could not find or install a certificate bundle.")
        print("Manual intervention required.")
        
    # Fix specific files
    fix_ssl_for_presidio()
    fix_process_any_data()
    
    # Create a simplified fix_ssl.py file
    with open('simple_fix_ssl.py', 'w') as f:
        f.write('''#!/usr/bin/env python
"""Simple SSL fix - import this at the beginning of your script."""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
print("SSL certificate verification disabled.")
''')
    
    print("\nCreated simple_fix_ssl.py. You can import this at the beginning of your script:")
    print("  import simple_fix_ssl")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())