import os
import shutil

old_path = " "
new_path = " "

for i in range(1, 901):
    file = os.path.join(old_path, f'{i}.jpg')
    for j in range(1, 15):
        newname = os.path.join(new_path, f'{i}_{j}' + '.jpg')
        shutil.copy(file, newname)
