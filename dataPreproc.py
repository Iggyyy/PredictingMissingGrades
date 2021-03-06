import json
import numpy as np

f=open("training.json")

data = list()
labels = list()
subjects = set()

#------READING FROM TRAINING FILE
i = 0
how_many_to_read=20000
for x in f:
    if i >= how_many_to_read: break
    i+=1

    record = json.loads(x)
    labels.append(record['Mathematics'])

    record.pop("Mathematics")

    record.pop("serial")

    for subj in record:
        subjects.add(subj)
    data.append(record)
    

subjects = list(subjects)
print("---DataCheck---")
print(len(subjects), subjects)
print(len(data), data[0])
print(len(labels), labels[0])
print("---DataLoaded---")

#CONVERTING DATA INTO TRUE TABLES
training_data = list()
for rec in data:
    arr = np.zeros(len(subjects))

    for i, d in rec.items():
        arr[subjects.index(i)] = d/8

    training_data.append(arr)

print(training_data[0], labels[0])


training_labels = list()
for l in labels:
    arr = np.zeros(8)





print("--Training data processed succesfully--")
"""
Return 2 values, t_data and t_labels
"""
def getTrainingData():
    return np.array(training_data[0:15000]), np.array(labels[0:15000])
def getTestingData():
    return np.array(training_data[15000:20000]), np.array(labels[15000:20000])

print(type(np.array(training_data[0:15000])))