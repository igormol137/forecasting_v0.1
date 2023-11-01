import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Walk-Forward validation
def walk_forward_optimization(data, window, horizon):
    
    unit_of_time_list = []
    performance_list = []
    
    # Initialize starting point of the window
    window_start = 0
    
    # Rolling window loop
    for window_end in range(window, len(data) - horizon):
        
        # Training & testing data
        train = data[window_start:window_end]
        test = data[window_end:(window_end + horizon)]
        
        # Model training
        model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
        
        # Reshape the train data
        X_train = train['time_scale'].values.reshape(-1,1)
        y_train = train['sum_quant_item'].values
        
        model.fit(X_train, y_train)
        
        # Make a prediction
        X_test = test['time_scale'].values.reshape(-1, 1)
        y_test = test['sum_quant_item'].values
        predictions = model.predict(X_test)
        
        # Compute error metric
        error = mean_squared_error(y_test, predictions)
        
        # Append results to lists
        unit_of_time = X_train[1][0] - X_train[0][0]
        performance_list.append(error)
        unit_of_time_list.append(unit_of_time)
        
        # Move the window
        window_start += 1

    return unit_of_time_list, performance_list

# if your window size is 10 and horizon is 1
unit_of_time_list, performance_list = walk_forward_optimization(training_data, 10, 1)

# Find the optimal unit of time
optimal_index = np.argmin(performance_list)
optimal_unit_of_time = unit_of_time_list[optimal_index]

print(f'The optimal unit of time is {optimal_unit_of_time}.')
