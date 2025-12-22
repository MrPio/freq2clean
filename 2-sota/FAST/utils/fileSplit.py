import platform

def fileSplit(filePath):

    if platform.system().lower() == 'windows':
        fileSplit = filePath.split("\\",maxsplit=-1)
    if platform.system().lower() == 'linux':
        fileSplit = filePath.split(r"/",maxsplit=-1)
    return fileSplit

if __name__ == '__main__':
    path = "G:\Files\DeepFlow\DeepFlow_algorithm_pytorch\data"
    fileSplit(path)