import matplotlib.pyplot as plt
import DataGen, PolyModel
from utils import plotting

degree = 10
num_datapoints = 50
num_test = 100
minimum = -1
maximum = 1
noise = 0.5 # std
max_capacity = 100

data_gen = DataGen.PolyDataGen(degree)
train_data = data_gen.generate_data(num_datapoints, noise, minimum, maximum)
test_data = data_gen.generate_data(num_test, noise, minimum=min(train_data[0]), maximum=max(train_data[0]))

plotting.plot_data(test_data, "Test")
plotting.plot_data(train_data, "Train")

capacities = list(range(1, max_capacity + 1, 5))
train_rmses = []
test_rmses = []

for i in range(len(capacities)): # n
    capacity = capacities[i]
    print('.', end='')
    model = PolyModel.PolyModel(capacity)
    model.fit(*train_data)
    train_rmse = model.score(*train_data)
    test_rmse = model.score(*test_data)
    train_rmses.append(train_rmse)
    test_rmses.append(test_rmse)
    if i%(len(capacities)//5) == 1:
        plotting.plot_fn(model.predict, min(train_data[0]), max(train_data[0]), f"capacity: {capacity}")
print()
# Choose Y limits
y_low = min(min(train_data[1]), min(train_data[1])*2)
y_high = max(train_data[1])*2
plt.ylim([y_low, y_high])
plt.show()

plt.plot(capacities, train_rmses, label='Train')
plt.plot(capacities, test_rmses, label='Test')
plt.legend(); plt.show()
