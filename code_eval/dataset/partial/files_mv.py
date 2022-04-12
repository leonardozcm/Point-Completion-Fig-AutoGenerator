import os
import shutil

dir_name = "02958343"

subnames = os.listdir(dir_name)

for name in subnames:
    shutil.move(os.path.join(dir_name,name,name+".pcd"), os.path.join(dir_name,name+'.pcd'))
    os.removedirs(os.path.join(dir_name,name))
