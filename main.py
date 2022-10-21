import os
import ai_pygame as ap

def getInput(path):
    dir = os.scandir(path)
    maps, nMaps = [], 0
    for entry in dir:
        if entry.is_file():
            nMaps += 1
    for mapID in range(nMaps):
        input_path = path + f'\\input{mapID + 1}.txt'
        maps.append(ap.read_file(input_path))
    return maps

def level_1(maps):
    script_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(script_path)
    new_abs_path = os.path.join(dir_path, 'output\\level_1')
    if not os.path.exists(new_abs_path):
        os.mkdir(new_abs_path)
    pass

def level_2(maps):
    pass

def advance(maps):
    pass

print(getInput('input\\advance'))
