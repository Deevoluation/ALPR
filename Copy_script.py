import os
import shutil

src = []
dst = []

os.chdir('test_images')
for f in os.listdir():
    src.append(f)

os.chdir('..')
if os.path.exists('Result') == False:
    os.mkdir('Result')

#print(len(src))
#print(src)
for name in src:
    shutil.copy2('test1_images/'+name,'Result')
