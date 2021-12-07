import matplotlib.pyplot as plt
import DataGen, IterativeSinModel
from utils import plotting
from scipy.interpolate import make_interp_spline
import numpy as np

degree = 1
max_num_datapoints = 200
num_test = 500
minimum = -1
maximum = 1
train_noise = 0. # std
test_noise = 0.
max_capacity = 1000
N_SKIP = 1
REPEAT = 1 # Repeat with same coefficients
EPSILON = 0.1

data_gen = DataGen.TrigDataGen()

noise_to_accuracy = {}
train_ns = list(range(10,200+1))
train_data = data_gen.generate_data(max_num_datapoints, train_noise, minimum, maximum)
test_data = data_gen.generate_data(num_test, test_noise, minimum=min(train_data[0]), maximum=max(train_data[0]))
all_capacities = []
all_test_rmses = []
all_train_rmses = []
all_adversarial_rmses = []
for idx in range(len(train_ns)):
    n_train = train_ns[idx]
    # train_data = (train_data_0[0][:train_n], train_data_0[1][:train_n])
    # plotting.plot_data(test_data, "Test")
    # plotting.plot_data(train_data, "Train")
    
    capacities = sorted(set(list(range(1,max_num_datapoints+5))+list(range(1, max_capacity + 1, N_SKIP))))
    train_rmses = []
    test_rmses = []
    adversarial_rmses = []
    model =  IterativeSinModel.IterativeSinModel(max_capacity + 1)
    model.feed(train_data[0])
    model.feed_test(test_data[0])

    for i in range(len(capacities)): # n
        capacity = capacities[i]
        print('.', end='')
        model.fit(*train_data, capacity, n_train=n_train)
        train_rmse = model.score_train(train_data[1], capacity, n_train=n_train)
        test_rmse = model.score_test(test_data[1], capacity)
        # adversarial_rmse = model.score_adversarial(test_data[1], capacity, EPSILON)
        train_rmses.append(train_rmse)
        test_rmses.append(test_rmse)
        # adversarial_rmses.append(adversarial_rmse)
        # if i%(len(capacities)//3) == 1:
        #     plotting.plot_fn(lambda x: model.predict(capacity, model.generate_features(x)), min(train_data[0]), max(train_data[0]), f"capacity: {capacity}")
    print()
    all_capacities.append(capacities)
    all_train_rmses.append(train_rmses)
    all_test_rmses.append(test_rmses)
    # all_adversarial_rmses.append(adversarial_rmses)
    # Choose Y limits
    # y_low = min(min(train_data[1]), min(train_data[1])*2)
    # y_high = max(train_data[1])*2
    # plt.ylim([y_low, y_high])
    # plt.show()
    # plt.clf()
    
    # plt.plot(capacities, train_rmses, label='Train')
    # plt.plot(capacities, test_rmses, label='Test')
    # plt.plot(capacities, adversarial_rmses, label='Adversarial')
    # plt.axvline(x=num_datapoints, color='r', linestyle='--')
    # plt.legend(); plt.show()
    # plt.plot(capacities, train_rmses, label='Train')
    # plt.plot(capacities, test_rmses, label='Test')
    # plt.plot(capacities, adversarial_rmses, label='Adversarial')
    # plt.xscale('log')
    # plt.axvline(x=num_datapoints, color='r', linestyle='--')
    # plt.legend(); plt.show()
    # plt.plot(capacities, train_rmses, label='Train')
    # plt.xscale('log')
    # plt.axvline(x=num_datapoints, color='r', linestyle='--')
    # plt.legend(); plt.show()
    # plt.plot(capacities, test_rmses, label='Test')
    # plt.plot(capacities, adversarial_rmses, label='Adversarial')
    # plt.xscale('log')
    # plt.axvline(x=num_datapoints, color='r', linestyle='--')
    # plt.legend(); plt.show()




plt.style.use("seaborn")
plt.xscale("log")
for idx in range(len(train_ns)):
    train_n= train_ns[idx]
    X_Y_Spline = make_interp_spline(all_capacities[idx], all_test_rmses[idx])
    X_ = np.linspace(min(all_capacities[idx]), max(all_capacities[idx]), 500)
    Y_ = X_Y_Spline(X_)
    # plt.plot(X_, Y_, label='{:.0e}'.format(train_n))
    plt.plot(X_, Y_, label='{}'.format(train_n))
    # plt.plot(all_capacities[idx], all_test_rmses[idx], label='{:.2e}'.format(test_noise))
plt.legend()
plt.show()

for idx in range(len(train_ns)):
    train_n= train_ns[idx]
    X_Y_Spline = make_interp_spline(all_capacities[idx], all_test_rmses[idx])
    X_ = np.linspace(min(all_capacities[idx]), max(all_capacities[idx]), 500)
    Y_ = X_Y_Spline(X_)
    # plt.plot(X_, Y_, label='{:.0e}'.format(train_n))
    plt.plot(X_, Y_, label='{}'.format(train_n))
    # plt.plot(all_capacities[idx], all_test_rmses[idx], label='{:.2e}'.format(test_noise))
plt.legend()
plt.show()

plt.xscale("log")
for idx in range(len(train_ns)):
    train_n = train_ns[idx]
    plt.plot(all_capacities[idx][:200], all_train_rmses[idx][:200], label=train_n)
plt.legend()
plt.show()

# # Plot Adversarial Change
# plt.xscale('log')
# for idx in range(len(test_noises)):
#     adversarial_rmses = all_adversarial_rmses[idx]
#     test_rmses = all_test_rmses[idx]
#     adversarial_effect = 100 * (np.array(adversarial_rmses) - np.array(test_rmses))
#     X_Y_Spline = make_interp_spline(all_capacities[idx], adversarial_effect)
#     X_ = np.linspace(min(all_capacities[idx]), max(all_capacities[idx]), 500)
#     Y_ = X_Y_Spline(X_)
#     plt.plot(X_, Y_, label='{:.0e}'.format(test_noise))
# plt.show()
