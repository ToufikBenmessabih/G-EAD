import os

a = "./data/breakfast/results/supervised_C2FTCN/split3/predict_breakfast\P29_cam01_P29_cereals.txt"
b = "./data/breakfast//groundTruth/"
print('file: ', a)
print('GT: ', b)
gt_file = b + os.path.basename(a)

print('final: ', gt_file)