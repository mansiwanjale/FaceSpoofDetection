import os   #for creating directories
import random   #generating random no. and shuffling data
import shutil   #cp and mv files
from itertools import islice        #slicing of iterators for dividing data into chunks

outputFolderPath = "Dataset/SplitData" #split data is stored here
inputFolderPath = "Dataset/all" #original unsplit data is stored
splitRatio = {"train": 0.7, "val": 0.2, "test": 0.1}    #ratio for splitting data
#70% training ; 20 %validation and 10% testing
classes = ["fake", "real"]

try:
    shutil.rmtree(outputFolderPath)
    # print("Removed Directory")
except OSError as e:
    os.mkdir(outputFolderPath)

# ------ Directories to create ------
#os.makedir creates directories recursively ......if already exists then error is not raised
os.makedirs(f"{outputFolderPath}/train/images", exist_ok = True)
os.makedirs(f"{outputFolderPath}/train/labels", exist_ok = True)
os.makedirs(f"{outputFolderPath}/val/images", exist_ok = True)
os.makedirs(f"{outputFolderPath}/val/labels", exist_ok = True)
os.makedirs(f"{outputFolderPath}/test/images", exist_ok = True)
os.makedirs(f"{outputFolderPath}/test/labels", exist_ok = True)

# ------ get the names ------
listNames = os.listdir(inputFolderPath)
#lists the name of the entries specified in the dir in inputFolderPath
# print(listNames)
# print(len(listNames))
uniqueNames = []    #store unique names of files
for name in listNames:
    uniqueNames.append(name.split('.')[0])
   # in inputFolderPath all the files are iterated over...and only names are taken
uniqueNames = list(set(uniqueNames))
#converted to set for removing duplicates

# ------ shuffle ------
random.shuffle(uniqueNames)

# ------ find the number of images for each folder ------
lenData = len(uniqueNames)  #represents an individual data sample
lenTrain = int(lenData * splitRatio['train'])   #calcs number of images in training set
lenVal = int(lenData * splitRatio['val'])
lenTest = int(lenData * splitRatio['test'])

# ------ put remaining images in training ------
if lenData != lenTrain + lenTest + lenVal:
    remaining = lenData - (lenTrain + lenTest + lenVal)
    lenTrain += remaining

# ------ split the list ------
lengthToSplit = [lenTrain, lenVal, lenTest]
Input = iter(uniqueNames)
Output = [list(islice(Input, elem)) for elem in lengthToSplit]
print(f'Total Images: {lenData} \nSpilt: {len(Output[0])} {len(Output[1])} {len(Output[2])}')

# ------ copy the files ------
sequence = ['train', 'val', 'test']
for i, out in enumerate(Output):
    for filename in out:
        shutil.copy(f'{inputFolderPath}/{filename}.jpg', f'{outputFolderPath}/{sequence[i]}/images/{filename}.jpg')
        shutil.copy(f'{inputFolderPath}/{filename}.txt', f'{outputFolderPath}/{sequence[i]}/labels/{filename}.txt')

print("Split Process Completed...")

# ------ creating Data.yaml file ------
dataYaml = f'path: ../Data\n\
^^^
#root path for the dataset  (Adjustable according to the dataset)
train: ../train/images\n\       #training images path
val: ../val/images\n\           #validation images path
test: ../test/images\n\         #test images path
\n\
nc: {len(classes)} \n\          #number of classes
names: {classes} '


f = open(f"{outputFolderPath}/data.yaml", 'a')  #opens the data.yaml file in append mode
f.write(dataYaml)
f.close()

print("Data.yaml file created...")