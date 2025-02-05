from datasets import load_from_disk, concatenate_datasets

# Replace these file paths with the paths to your local datasets.
# In this example, we assume the datasets are stored as CSV files.
path_dataset1 = "data/s1"
path_dataset2 = "data/s2"

# Load the first dataset.
# The `split="train"` argument extracts the default split.
dataset1 = load_from_disk(dataset_path=path_dataset1)

# Load the second dataset.
dataset2 = load_from_disk(dataset_path=path_dataset2)

# Optionally, check that both datasets have the same features (columns)
if dataset1.features != dataset2.features:
    raise ValueError("The two datasets do not have the same structure!")
    
# Merge (concatenate) the two datasets.
merged_dataset = concatenate_datasets([dataset1, dataset2])

# sample to train split and validation split
merged_dataset = merged_dataset.train_test_split(test_size=0.1)

# Save the merged dataset locally.
# This will create a directory named "merged_dataset" containing the saved dataset.
merged_dataset.save_to_disk("data/merged")

print("Datasets merged and saved successfully!")
