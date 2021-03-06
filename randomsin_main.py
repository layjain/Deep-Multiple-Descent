import matplotlib.pyplot as plt
import DataGen, RandomSinModel
from utils import plotting

degree = 1
num_datapoints = 100
num_test = 500
minimum = -1
maximum = 1
train_noise = 0. # std
test_noise = 0.
max_capacity = 1000
N_SKIP = 3
REPEAT = 1 # Repeat with same coefficients

data_gen = DataGen.TrigDataGen()

for _ in range(REPEAT):
    train_data = data_gen.generate_data(num_datapoints, train_noise, minimum, maximum)
    test_data = data_gen.generate_data(num_test, test_noise, minimum=min(train_data[0]), maximum=max(train_data[0]))
    
    plotting.plot_data(test_data, "Test")
    plotting.plot_data(train_data, "Train")
    
    capacities = sorted(set(list(range(1,num_datapoints+5))+list(range(1, max_capacity + 1, N_SKIP))))
    train_rmses = []
    test_rmses = []
    
    for i in range(len(capacities)): # n
        capacity = capacities[i]
        print('.', end='')
        model = RandomSinModel.RandomSinModel(capacity)
        model.fit(*train_data)
        train_rmse = model.score(*train_data)
        test_rmse = model.score(*test_data)
        train_rmses.append(train_rmse)
        test_rmses.append(test_rmse)
        if i%(len(capacities)//3) == 1:
            plotting.plot_fn(model.predict, min(train_data[0]), max(train_data[0]), f"capacity: {capacity}")
    print()
    # Choose Y limits
    y_low = min(min(train_data[1]), min(train_data[1])*2)
    y_high = max(train_data[1])*2
    plt.ylim([y_low, y_high])
    # plt.show()
    plt.clf()
    
    plt.plot(capacities, train_rmses, label='Train')
    plt.plot(capacities, test_rmses, label='Test')
    plt.axvline(x=num_datapoints, color='r', linestyle='--')
    plt.legend(); plt.show()
    plt.plot(capacities, train_rmses, label='Train')
    plt.plot(capacities, test_rmses, label='Test')
    plt.xscale('log')
    plt.axvline(x=num_datapoints, color='r', linestyle='--')
    plt.legend(); plt.show()
    plt.plot(capacities, train_rmses, label='Train')
    plt.xscale('log')
    plt.axvline(x=num_datapoints, color='r', linestyle='--')
    plt.legend(); plt.show()
