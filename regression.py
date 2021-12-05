import torch
from torch.autograd import Variable
import DataGen, IterativeSinModel
import matplotlib.pyplot as plt
import numpy as np

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

outputDim = 1       # takes variable 'y'
learningRate = 0.1 
epochs = 500

num_datapoints = 10
num_test = 500
minimum = -1
maximum = 1
train_noise = 0.0 # std
test_noise = 0.
max_capacity = 50
N_SKIP = 2
REPEAT = 30

data_gen = DataGen.TrigDataGen()
train_data = data_gen.generate_data(num_datapoints, train_noise, minimum, maximum)
test_data = data_gen.generate_data(num_test, test_noise, minimum=min(train_data[0]), maximum=max(train_data[0]))

tranformation =  IterativeSinModel.IterativeSinModel(max_capacity + 1)
tranformation.feed(train_data[0])
tranformation.feed_test(test_data[0])

criterion = torch.nn.MSELoss() 

capacities = sorted(set(list(range(1,num_datapoints+5))+list(range(1, max_capacity + 1, N_SKIP))))
# capacities = [10]
train_losses = []
test_losses = []
for c in capacities: 
    print(f"Fitting Curve for capacity {c}")
    # Converting inputs and labels to Variable
    train_inputs = Variable(torch.from_numpy(tranformation.generate_features(train_data[0])[:,:c])).float()
    train_labels = Variable(torch.from_numpy(train_data[1])).float()

    test_inputs = Variable(torch.from_numpy(tranformation.generate_features(test_data[0])[:,:c])).float()
    test_labels = Variable(torch.from_numpy(test_data[1])).float()

    train_loss_capacity = []
    test_loss_capacity = []
    min_norm = float('inf')
    min_norm_run = None
    for r in range(REPEAT):
        model = linearRegression(c, outputDim)
        optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
        
        for epoch in range(epochs):

            # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
            optimizer.zero_grad()

            # get output from the model, given the inputs
            outputs = model(train_inputs)

            # get loss for the predicted output
            loss = criterion(outputs, train_labels)
            # print(loss)
            # get gradients w.r.t to parameters
            loss.backward()

            # update parameters
            optimizer.step()
            # print('epoch {}, loss {}'.format(epoch, loss.item()))
        
        train_loss_capacity.append(loss.item())

        # calculate_test_loss
        with torch.no_grad(): # we don't need gradients in the testing phase
            predicted = model(test_inputs)
            loss = criterion(predicted, test_labels)
            test_loss_capacity.append(loss.item())
        
        weight_norm = torch.norm(model.linear.weight)
        if weight_norm<min_norm:
            min_norm_run = r
            min_norm = weight_norm

    train_losses.append(train_loss_capacity[r])
    test_losses.append(test_loss_capacity[r])

# print(train_losses, test_losses)
plt.plot(capacities, train_losses, label="Train")
plt.plot(capacities, test_losses, label="Test")
plt.legend()
plt.show()