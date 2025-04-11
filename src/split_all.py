"""
This script runs all the split scripts in the src folder.
"""
import importlib
split_scripts = [
    # 'random_splitter',
    'random_splitter_voiced',
    # 'each_patient_random_segment_splitter',
    'each_patient_random_segment_splitter_voiced',
    # 'patient_wise_splitter',
    'patient_wise_splitter_voiced',
    # 'each_patient_same_segment_splitter',
    'each_patient_same_segment_splitter_voiced'
]

for script in split_scripts:
    print(f"Running {script}.py...")
    importlib.__import__(f"{script}")
