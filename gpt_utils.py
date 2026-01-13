import codecs, os, ast, json
from re import finditer
import pandas as pd
import shutil

def read_file(file_path):
    with codecs.open(file_path, "r", encoding='utf-8') as file:
        return file.read()

def write_text(file_name, method, text):
    """
    write text to file
    method: 'a'-append, 'w'-overwrite
    """
    with open(file_name, method, encoding='utf-8') as f:
        f.write(text + '\n')

def convertStrToDict(file_cont):
    return ast.literal_eval(file_cont)

def convertDictToStr(src_dict):
    return json.dumps(src_dict)

def removeFileIfExists(file_path):
    if os.path.isfile(file_path):
        os.remove(file_path)
def camel_case_split(identifier):
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0).lower() for m in matches]

def getPickleData(pickle_load_path):
    """
    Load a pickle file as a Pandas dataframe object.
    """
    obj = pd.read_pickle(pickle_load_path)
    return obj

def savePickleData(pickle_obj, pickle_save_path):
    """
    Save Pandas dataframe object as a pickle file.
    """
    df = pd.DataFrame(pickle_obj)
    df.to_pickle(pickle_save_path)

def cleanPickleWithIndex(src_obj, index_list):
    """
    Cleaning a pickle object with a list of index.
    The indices from the list will be deleted but the existing indices still remain as they are.
    For example, [1, 2, 3, 4, 5, 6, 7] - [2, 3, 5, 6] = [1, 4, 7]
    """
    dst_obj = src_obj.drop(labels=index_list, axis=0)
    return dst_obj

def listIntersection(list_1, list_2):
    """
    Input: two lists
    Output: an intersected list
    """
    set1 = set(list_1)
    set2 = set(list_2)
    newList = list(set1.intersection(set2))
    print("Intersection of the lists is:", newList)
    return newList

def clearDirectory(folder):
    if os.path.isdir(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    else:
        os.makedirs(folder)
