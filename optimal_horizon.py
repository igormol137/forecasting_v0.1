from sklearn.metrics import mean_squared_error
import numpy as np

def walk_forward_optimization(training_data, min_horizon, max_horizon, step, window):
    
    num_instances = len(training_data)
    instances_horizon_rmse = []
    
    for horizon in range(min_horizon, max_horizon+1, step):
        horizon_errors = []
        for i in range(window, num_instances - horizon):
            train = training_data[i-window:i]
            test = training_data[i:i+horizon]
      
            model = train_model(train)
            predictions = model.predict(test['time_scale'].values.reshape(-1,1))
            
            mse = mean_squared_error(test['sum_quant_item'], predictions)
            horizon_errors.append(mse)
            
        instances_horizon_rmse.append((horizon, np.mean(horizon_errors)))

    instances_horizon_rmse.sort(key=lambda tup: tup[1])  # Sort by RMSE
    best_instance, best_error = instances_horizon_rmse[0]
    print("Optimal horizon is: %d with RMSE of %.3f " % (best_instance, best_error))
    
    return best_instance

# Assuming that training_data is your DataFrame

# Define minimum and maximum horizons range (e.g., from 1 to 10)
min_horizon = 1
max_horizon = 10

# Define the step size for the horizon (e.g., step = 1)
step_size = 1

# Define the window size for training the SVR model (e.g., window = 100)
window_size = 100

# Applying the walk forward optimization
optimal_horizon = walk_forward_optimization(training_data, min_horizon, max_horizon, step_size, window_size)
