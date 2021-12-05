import matplotlib.pyplot as plt
import DataGen, IterativeSinModel
from utils import plotting

degree = 1
num_datapoints = 20
num_test = 500
minimum = -1
maximum = 1
train_noise = 0.1 # std
test_noise = 0.
max_capacity = 2000
N_SKIP = 5
REPEAT = 1 # Repeat with same coefficients
EPSILON = 0.1

data_gen = DataGen.TrigDataGen()

for _ in range(REPEAT):
    train_data = data_gen.generate_data(num_datapoints, train_noise, minimum, maximum)
    test_data = data_gen.generate_data(num_test, test_noise, minimum=min(train_data[0]), maximum=max(train_data[0]))
    
    plotting.plot_data(test_data, "Test")
    plotting.plot_data(train_data, "Train")
    
    capacities = sorted(set(list(range(1,num_datapoints+5))+list(range(1, max_capacity + 1, N_SKIP))))
    train_rmses = []
    test_rmses = []
    adversarial_rmses = []
    model =  IterativeSinModel.IterativeSinModel(max_capacity + 1)
    model.feed(train_data[0])
    model.feed_test(test_data[0])
    thetas = {}
    
    for i in range(len(capacities)): # n
        capacity = capacities[i]
        print('.', end='')
        model.fit(*train_data, capacity)
        train_rmse = model.score_train(train_data[1], capacity)
        test_rmse = model.score_test(test_data[1], capacity)
        adversarial_rmse = model.score_adversarial(train_data[1], capacity, EPSILON)
        train_rmses.append(train_rmse)
        test_rmses.append(test_rmse)
        adversarial_rmses.append(adversarial_rmse)
        thetas[capacity] = model.a
        if i%(len(capacities)//3) == 1:
            plotting.plot_fn(lambda x: model.predict(capacity, model.generate_features(x)), min(train_data[0]), max(train_data[0]), f"capacity: {capacity}")
    print()
    # Choose Y limits
    y_low = min(min(train_data[1]), min(train_data[1])*2)
    y_high = max(train_data[1])*2
    plt.ylim([y_low, y_high])
    plt.show()
    plt.clf()
    
    plt.plot(capacities, train_rmses, label='Train')
    plt.plot(capacities, test_rmses, label='Test')
    plt.plot(capacities, adversarial_rmses, label='Adversarial')
    plt.axvline(x=num_datapoints, color='r', linestyle='--')
    plt.legend(); plt.show()
    plt.plot(capacities, train_rmses, label='Train')
    plt.plot(capacities, test_rmses, label='Test')
    plt.plot(capacities, adversarial_rmses, label='Adversarial')
    plt.xscale('log')
    plt.axvline(x=num_datapoints, color='r', linestyle='--')
    plt.legend(); plt.show()
    plt.plot(capacities, train_rmses, label='Train')
    plt.xscale('log')
    plt.axvline(x=num_datapoints, color='r', linestyle='--')
    plt.legend(); plt.show()
    plt.plot(capacities, test_rmses, label='Test')
    plt.xscale('log')
    plt.axvline(x=num_datapoints, color='r', linestyle='--')
    plt.legend(); plt.show()
    plt.plot(capacities, adversarial_rmses, label='Adversarial')
    plt.xscale('log')
    plt.axvline(x=num_datapoints, color='r', linestyle='--')
    plt.legend(); plt.show()

