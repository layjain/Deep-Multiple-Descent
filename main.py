import data, PolyModel

degree = 2
num_datapoints = 10
minimum = -1
maximum = 1
noise = 0.1 # std
max_capacity = 50

train_data = data.generate_data(degree, num_datapoints, noise, minimum, maximum)
test_data = data.generate_data(degree, num_datapoints, noise, minimum, maximum)

capacities = list(range(1, max_capacity + 1))
train_rmses = []
test_rmses = []

for capacity in capacities: # n
    print('.', end='')
    model = PolyModel.PolyModel(capacity)
    model.fit(*train_data)
    train_rmse = model.score(*train_data)
    test_rmse = model.score(*test_data)
    train_rmses.append(train_rmse)
    test_rmses.append(test_rmse)

import matplotlib.pyplot as plt

plt.plot(capacities, train_rmses, label='Train')
plt.plot(capacities, test_rmses, label='Test')
plt.legend(); plt.show()
