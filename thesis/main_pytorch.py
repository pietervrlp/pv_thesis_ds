import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from utils import process_dataframe, plot_autocorrelations, split_train_test, predict_index, get_X_y, scaler_function 
from utils import real_predicted_results, get_test_plot, mean_absolute_percentage_error_u
from gan_model.cnn_bigru_pytorch import Generator, Discriminator, WGAN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    with open("main_config.yml") as file:
            config = yaml.safe_load(file)

    # Read the data
    dataframe = pd.read_feather(config['dataset'])

    #Process the data
    df = process_dataframe(dataframe)

    # Separate X variables and y variable (closing price)
    X_value = pd.DataFrame(df.iloc[:, :])
    y_value = pd.DataFrame(df.iloc[:, 0]) #first column is the closing price

    # Lags check (to have an idea about autocorrelation)
    # plot_autocorrelations(y_value)

    # Split train and test data
    n_steps_in = config['n_steps_in']
    n_features = X_value.shape[1]
    n_steps_out = config['n_steps_out']

    X_train, X_test = split_train_test(X_value, train_dimension=config['train_dimension'])
    y_train, y_test = split_train_test(y_value.values, train_dimension=config['train_dimension'])

    # Normalize the data
    X_scaler, y_scaler = scaler_function(X_train, y_train)
    
    X_train = X_scaler.fit_transform(X_train)
    y_train = y_scaler.fit_transform(y_train)
    X_test = X_scaler.fit_transform(X_test)
    y_test = y_scaler.fit_transform(y_test)

    # Reshape the data as input for the model
    X_train, y_train, past_y_train = get_X_y(X_train, y_train, n_steps_in, n_steps_out)
    X_test, y_test, past_y_test = get_X_y(X_test, y_test, n_steps_in, n_steps_out)
    index_train, index_test = predict_index(df, X_train, n_steps_in, n_steps_out)

    #WGAN-GP
    input_dim = X_train.shape[1]
    feature_size = X_train.shape[2]
    output_dim = y_train.shape[1]
    epoch = config["epochs"]

    generator = Generator(input_dim=input_dim, output_dim=output_dim, feature_size=feature_size).to(device)
    discriminator = Discriminator(n_steps_in=n_steps_in, n_steps_out=n_steps_out).to(device)
    wgan = WGAN(generator=generator, discriminator=discriminator, n_steps_in=n_steps_in, n_steps_out=n_steps_out)

    X_train = torch.from_numpy(X_train).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    past_y_train = torch.from_numpy(past_y_train).float().to(device)

    Predicted_price, Real_price = wgan.train(X_train, y_train, past_y_train, epoch)

    # Rescale to original values
    rescaled_Real_price = y_scaler.inverse_transform(Real_price.cpu().numpy())
    rescaled_Predicted_price = y_scaler.inverse_transform(Predicted_price.cpu().numpy())

    # Predicted results and real value
    predict_result, real_price = real_predicted_results(rescaled_Predicted_price, rescaled_Real_price, index_train, output_dim)
    predict_result['predicted_mean'] = predict_result.mean(axis=1)
    real_price['real_mean'] = real_price.mean(axis=1)

    # Plot the predicted result
    plt.figure(figsize=(16, 8))
    plt.plot(real_price["real_mean"])
    plt.plot(predict_result["predicted_mean"], color='r')
    plt.xlabel("Date")
    plt.ylabel("Stock price")
    plt.legend(("Real price", "Predicted price"), loc="upper left", fontsize=16)
    plt.title("Training set", fontsize=20)
    plt.savefig('./outcome/train_predictions.png')

    # Calculate RMSE, MAE, and MAPE
    predicted = predict_result["predicted_mean"]
    real = real_price["real_mean"]
    RMSE = np.sqrt(nn.MSELoss(predicted, real))
    MAE = nn.L1Loss(predicted, real)
    MAPE = mean_absolute_percentage_error_u(predicted, real)

    print('Train RMSE: ', RMSE)
    print('Train MAE: ', float(MAE))
    print('Train MAPE: ', "{0:.5f}%".format(MAPE))

    # Test performance
    generator_model = torch.load(f'gen_model_{n_steps_in}_{n_steps_out}_{epoch-1}.pth')
    test_predicted, test_RMSE, test_MAE, test_MAPE = get_test_plot(generator_model, X_test, y_test, y_scaler, index_test, output_dim)   