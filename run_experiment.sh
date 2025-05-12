#!/bin/bash
# Run the PII/PHI detection experiment with reasonable defaults

# Generate synthetic data
python run.py --generate_data --num_documents 100 --run_genetic --population_size 50 --generations 20

# Check if experiment was successful
if [ $? -eq 0 ]; then
    echo "Experiment completed successfully!"
    echo "Results are in the 'results' directory."
else
    echo "Experiment failed. Check 'experiment.log' for details."
fi