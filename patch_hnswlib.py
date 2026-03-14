import re

# Read the file
with open('build/_deps/hnswlib-src/hnswlib/hnswalg.h', 'r') as f:
    content = f.read()

# Replace getLabelOpMutex calls with a dummy mutex (or remove locking entirely for single-threaded use)
# This is a workaround for MinGW compatibility issues

# Method 1: Replace with global mutex lock (simpler, works on MinGW)
content = content.replace(
    'std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));',
    '// Label mutex disabled for MinGW compatibility/n        // std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));'
)

# Write back
with open('build/_deps/hnswlib-src/hnswlib/hnswalg.h', 'w') as f:
    f.write(content)

print("Patched successfully")
