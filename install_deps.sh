#!/bin/bash
# Install required dependencies for PHI detection

echo "Installing Python dependencies..."
pip install PyPDF2 pdfplumber tqdm pandas numpy

# Only install these packages if explicitly requested (they're larger)
if [[ "$1" == "--full" ]]; then
    echo "Installing additional dependencies for advanced features..."
    pip install scikit-learn deap transformers torch textract
fi

echo "Checking if installation was successful..."

# Check if PDF libraries are available
python -c "import PyPDF2; print('PyPDF2 installed successfully')" || echo "WARNING: PyPDF2 installation failed"
python -c "import pdfplumber; print('pdfplumber installed successfully')" || echo "WARNING: pdfplumber installation failed"

echo "Installation complete. You can now run:"
echo "./process_any_data.py --input <your_data_directory> --output <results_directory>"