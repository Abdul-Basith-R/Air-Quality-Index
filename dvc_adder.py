import os
from glob import glob


files = glob(os.path.join("data", "raw","air_data", "*.csv"))
# print(files)
for filePath in files:
    # print(filePath)
    # print(f"dvc add {filePath}")
    os.system(f"dvc add {filePath}")


print("\n #### all files added to dvc ####")
