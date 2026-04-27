import os

# Get files directory contents and sort for consistent ordering
files_dir = "files"
filenames = sorted(os.listdir(files_dir))

# Filter to only process actual files (not directories)
filenames = [f for f in filenames if os.path.isfile(os.path.join(files_dir, f))]

for i, filename in enumerate(filenames):
    old_path = os.path.join(files_dir, filename)
    new_path = os.path.join(files_dir, f"report_{i}.txt")
    
    # Avoid overwriting if target already exists
    if not os.path.exists(new_path):
        os.rename(old_path, new_path)
        print(f"Renamed {filename} -> report_{i}.txt")
    else:
        print(f"Skipped {filename}: report_{i}.txt already exists")