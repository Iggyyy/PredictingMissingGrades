import numpy as np
import torch
import dataPreproc as dp
torch.manual_seed(1)
cuda = torch.device('cuda')
print(cuda)

raw_train_x, raw_train_y = dp.getTrainingData()
raw_test_x, raw_test_y = dp.getTestingData()

INPUT_SIZE = len(raw_train_x[0])
HIDDEN_SIZE = 20
OUT_SIZE = 8
ITERATIONS = 50 
print(np.shape(raw_train_x))

train_x = torch.Tensor(raw_train_x).cuda()
train_y = torch.Tensor(raw_train_y).cuda()

test_x = torch.Tensor(raw_test_x).cuda()
test_y = torch.Tensor(raw_test_y).cuda()

#CREATING LAYERS CONNECTED TO FIXED SIZES
lay_0 = torch.nn.Linear(INPUT_SIZE, HIDDEN_SIZE).cuda()
lay_1 = torch.nn.Linear(HIDDEN_SIZE, OUT_SIZE).cuda()

model = torch.nn.Sequential(lay_0,  torch.nn.Dropout(p=0.4), torch.nn.ReLU(), lay_1, torch.nn.Softmax() ).cuda()
criterion = torch.nn.MSELoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for i in range(ITERATIONS):
    prediction = model.forward(train_x).cuda()

    loss = criterion.forward(prediction, train_y).cuda()

    loss.backward(torch.ones_like(loss.data))

    optimizer.step()


    if i % 25 == 0:
        t_prediction = model.forward(test_x).cuda()
        correct_cnt = 0
        for k in range(len(t_prediction)):
            correct_cnt += int( torch.argmax(t_prediction[k]) ==torch.argmax(test_y[k]) )
        
        print("Iteration:" + str(i) + " MSE: ",  loss,  "Test correct percentage: " + str(100*correct_cnt/len(t_prediction)) + "%")



