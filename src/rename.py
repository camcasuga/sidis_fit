import os
import re

def main(i, dir):
    file_name = f"{i}.dat"
    path_to_file = os.path.join(dir, file_name)
    file = open(path_to_file, 'r')
    string_to_find = "# Parameter file of fit of dipole from"
    
    for f in file.readlines():
        if string_to_find in f:
            print(f)
            num = int(re.search(r'\d+', f).group())
            print(num)

    new_file_name = f"fp_{num}.dat"
    new_path_to_file = os.path.join(dir, new_file_name)

    os.rename(path_to_file, new_path_to_file)
    


if __name__ == "__main__":
    for i in range(0,100):
        main(i, "../data/dipole/fit_params/")

