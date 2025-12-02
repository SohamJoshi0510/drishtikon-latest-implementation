import os
import re

def create_requirements(file):
    with open(file, 'r') as fileObject:
        lines = fileObject.readlines()
        cleaned = "".join(lines).split('\n')[2:]
        res_list = []
        for line in cleaned:
            res = ''
            space_enc = False
            for char in line:
                if not char.strip():
                    if not space_enc:
                        res += "=="
                    space_enc = True
                    continue
                res += char
            res_list.append(res)
    name, ext = os.path.splitext(file)
    count = 0
    outputFilepath = f'{name}{"_" * count}{ext}'
    while os.path.exists(outputFilepath):
        count += 1
        outputFilepath = f'{name}{"_" * count}{ext}'
        
    with open(outputFilepath, 'a') as fileObj:
        fileObj.writelines("\n".join(res_list))

def listChildren(filepath):
    for childFileBasename in os.listdir(path=filepath):
        absolutePath = os.path.join(filepath, childFileBasename)
        if os.path.isdir(absolutePath):
            listChildren(absolutePath)
            global count
            count -= 1
        else:
            print(absolutePath)
    count += 1

import os
create_requirements(os.path.join(os.getcwd(), 'requirements.txt'))