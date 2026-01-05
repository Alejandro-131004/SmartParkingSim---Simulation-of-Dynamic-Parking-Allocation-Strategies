import sys
import os

print("Python Executable:", sys.executable)
print("System Path:")
for p in sys.path:
    print(p)

try:
    import yaml
    print("PyYAML imported successfully rulez")
except ImportError as e:
    print("ImportError:", e)

try:
    import pandas
    print("Pandas imported successfully")
except ImportError as e:
    print("ImportError:", e)
