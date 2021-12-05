import matplotlib.pyplot as plt
import DataGen, IterativeSinModel
from utils import plotting
from scipy.interpolate import make_interp_spline
import numpy as np

degree = 1
num_datapoints = 20
num_test = 500
minimum = -1
maximum = 1
train_noise = 0. # std
test_noise = 0.
max_capacity = 2000
N_SKIP = 1
REPEAT = 1 # Repeat with same coefficients

data_gen = DataGen.TrigDataGen()

noise_to_accuracy = {}
test_noises = [0., 1e-5, 1e-1, 1e0]
train_data = data_gen.generate_data(num_datapoints, train_noise, minimum, maximum)
test_data_0 = data_gen.generate_data(num_test, test_noise, minimum=min(train_data[0]), maximum=max(train_data[0]))
all_capacities = []
all_test_rmses = []
all_train_rmses = []
for test_noise_idx in range(len(test_noises)):
    test_noise = test_noises[test_noise_idx]
    test_data = (test_data_0[0], data_gen.add_noise(test_data_0[1], test_noise))
    # plotting.plot_data(test_data, "Test")
    # plotting.plot_data(train_data, "Train")
    
    capacities = sorted(set(list(range(1,num_datapoints+5))+list(range(1, max_capacity + 1, N_SKIP))))
    train_rmses = []
    test_rmses = []
    model =  IterativeSinModel.IterativeSinModel(max_capacity + 1)
    model.feed(train_data[0])
    model.feed_test(test_data[0])
    
    for i in range(len(capacities)): # n
        capacity = capacities[i]
        print('.', end='')
        model.fit(*train_data, capacity)
        train_rmse = model.score_train(train_data[1], capacity)
        test_rmse = model.score_test(test_data[1], capacity)
        train_rmses.append(train_rmse)
        test_rmses.append(test_rmse)
        # if i%(len(capacities)//3) == 1:
        #     plotting.plot_fn(lambda x: model.predict(capacity, model.generate_features(x)), min(train_data[0]), max(train_data[0]), f"capacity: {capacity}")
    print()
    all_capacities.append(capacities)
    all_train_rmses.append(train_rmses)
    all_test_rmses.append(test_rmses)
    # Choose Y limits
    # y_low = min(min(train_data[1]), min(train_data[1])*2)
    # y_high = max(train_data[1])*2
    # plt.ylim([y_low, y_high])
    # plt.show()
    # plt.clf()
    
    # plt.plot(capacities, train_rmses, label='Train')
    # plt.plot(capacities, test_rmses, label='Test')
    # plt.axvline(x=num_datapoints, color='r', linestyle='--')
    # plt.legend(); plt.show()
    # plt.plot(capacities, train_rmses, label='Train')
    # plt.plot(capacities, test_rmses, label='Test')
    # plt.xscale('log')
    # plt.axvline(x=num_datapoints, color='r', linestyle='--')
    # plt.legend(); plt.show()
    # plt.plot(capacities, train_rmses, label='Train')
    # plt.xscale('log')
    # plt.axvline(x=num_datapoints, color='r', linestyle='--')
    # plt.legend(); plt.show()
    # plt.plot(capacities, test_rmses, label='Test')
    # plt.xscale('log')
    # plt.axvline(x=num_datapoints, color='r', linestyle='--')
    # plt.legend(); plt.show()




plt.style.use("seaborn")
plt.xscale("log")
for idx in range(len(test_noises)):
    test_noise = test_noises[idx]
    X_Y_Spline = make_interp_spline(all_capacities[idx], all_test_rmses[idx])
    X_ = np.linspace(min(all_capacities[idx]), max(all_capacities[idx]), 500)
    Y_ = X_Y_Spline(X_)
    plt.plot(X_, Y_, label='{:.0e}'.format(test_noise))
    # plt.plot(all_capacities[idx], all_test_rmses[idx], label='{:.2e}'.format(test_noise))
plt.legend()
plt.show()

plt.xscale("log")
for idx in range(len(test_noises)):
    test_noise = test_noises[idx]
    plt.plot(all_capacities[idx], all_train_rmses[idx], label='{:.2e}'.format(test_noise))
plt.legend()
plt.show()
