import os
import glob

# Locate the quaternions.py file
quaternions_file = None
for path in glob.glob("/usr/lib/python*/dist-packages/transforms3d/quaternions.py"):
    quaternions_file = path
    break

if not quaternions_file:
    raise FileNotFoundError("Could not find 'quaternions.py' in 'transforms3d'.")

print(f"Found quaternions.py at: {quaternions_file}")

# Read the file content
with open(quaternions_file, "r") as file:
    lines = file.readlines()

# Flags to ensure only the first occurrences are updated
max_float_updated = False
float_eps_updated = False

# Update _MAX_FLOAT and _FLOAT_EPS lines
for i, line in enumerate(lines):
    if not max_float_updated and "_MAX_FLOAT" in line:
        lines[i] = "_MAX_FLOAT = np.float64\n"
        max_float_updated = True
    if not float_eps_updated and "_FLOAT_EPS" in line:
        lines[i] = "_FLOAT_EPS = np.finfo(np.float64).eps\n"
        float_eps_updated = True

    # Break early if both updates are done
    if max_float_updated and float_eps_updated:
        break

# Write back the modified file
with open(quaternions_file, "w") as file:
    file.writelines(lines)

print(f"Updated {quaternions_file} successfully!")
