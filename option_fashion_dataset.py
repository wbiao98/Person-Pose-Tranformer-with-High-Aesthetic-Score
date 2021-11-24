import os
import shutil
from PIL import Image


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in  IMG_EXTENSIONS)

def make_dataset(dir):
    image = []

    train_root = os.path.join(dir,'train')
    if not os.path.exists(train_root):
        os.mkdir(train_root)

    test_root = os.path.join(dir,'test')
    if not os.path.exists(test_root):
        os.mkdir(test_root)

    train_images = []
    train_file = open(os.path.join(dir,'train.lst'),'r')
    for lines in  train_file:
        lines = lines.strip()
        if lines.endswith('.jpg'):
            train_images.append(lines)

    test_images = []
    test_file = open(os.path.join(dir,'test.lst'),'r')
    for lines in  test_file:
        lines = lines.strip()
        if lines.endswith('.jpg'):
            test_images.append(lines)
       
    for root, _, filenames in sorted(os.walk(os.path.join(dir, 'img_highres'))):
        for filename in filenames:
            if is_image_file(filename):
                path = os.path.join(root,filename)
                path_name = path.split("/")
                print(path_name)

                path_names = path_name[2:]
                path_names[0] = 'fashion'
                path_names[3] = path_names[3].replace('_','')
                path_names[4] = path_names[4].split('_')[0] + "_" + "".join(path_names[4].split('_')[1:])
                path_names = "".join(path_names)
                if path_names in train_images:
                    shutil.copy(path,os.path.join(train_root,path_names))
                    print(os.path.join(train_root,path_names))
                    pass
                elif path_names in test_images:
                    shutil.copy(path,os.path.join(test_root,path_names))
                    print(os.path.join(test_root,path_names))
                    pass
def test_make_dataset(dir):
    for root, _, filenames in sorted(os.walk(os.path.join(dir, 'img_highres'))):
        for filename in filenames:
            if is_image_file(filename):
                path = os.path.join(root,filename)
                path_name = path.split("/")
                print(path_name)
                path_names = path_name[2:]
                print(path_names)
                path_names[0] = 'fashion'
                print(path_names)
                path_names[3] = path_names[3].replace('_','')
                print(path_names)
                path_names[4] = path_names[4].split('_')[0] + "_" + "".join(path_names[4].split('_')[1:])
                print(path_names)
                path_names = "".join(path_names)
                print(path_names)
make_dataset('./dataset/')