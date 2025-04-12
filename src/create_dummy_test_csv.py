import os
import csv

# Set the paths (update these as needed)
test_data_dir = "../data/test/"
dummy_test_csv = "../data/dummy_test.csv"
train_csv = "../data/train.csv"  # Used only to extract the header

# Read the header from train.csv
with open(train_csv, 'r', newline='') as f:
    reader = csv.reader(f)
    header = next(reader)  # e.g., ["filename", "label"]

# List all .aiff files in the test directory
test_files = [f for f in os.listdir(test_data_dir) if f.lower().endswith('.aiff')]

# Write the dummy CSV file
with open(dummy_test_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for file in test_files:
        # Here, we assign a dummy label, e.g., 0, for every test file.
        writer.writerow([file, 0])

print("Dummy test CSV file created at:", dummy_test_csv)