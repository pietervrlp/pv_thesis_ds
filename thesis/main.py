import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.losses import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from keras.initializers import RandomNormal
from keras import backend as K

from utils import process_dataframe, split_train_test, predict_index, get_X_y, scaler_function 
from utils import real_predicted_results, get_test_plot
from gan_model.cnn_bigru import Generator, Discriminator, WGAN     # Specify the desired GAN model to be imported here


# Session & seed configurations
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

tf.keras.utils.set_random_seed(1)

sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)

if __name__ == '__main__':
    with open("main_config.yml") as file:
            config = yaml.safe_load(file)

    data = config['data']
    stocks =  config['stocks'] #['ASTE', 'CENX', 'GIFI', 'HOLX', 'IBOC', 'NTRS', 'PRGS']

    for stock in stocks:
        trial_id = f'df_{stock}_{data}_cnn_bigru'
        dataframe = pd.read_feather(f'./data/{data}/df_{stock}_{data}.feather')

        # Process the data
        df = process_dataframe(dataframe)

        # Separate X variables and y variable
        X_value = pd.DataFrame(df.iloc[:, :])
        y_value = pd.DataFrame(df.iloc[:, 0]) # first column is the close price

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

        # GAN
        weight_initializer = RandomNormal(mean=0.00, stddev=0.02)

        input_dim = X_train.shape[1]
        feature_size = X_train.shape[2]
        output_dim = y_train.shape[1]
        epoch = config["epochs"]

        generator = Generator(X_train.shape[1], output_dim, X_train.shape[2], weight_initializer)
        discriminator = Discriminator(weight_initializer, n_steps_in, n_steps_out)
        wgan = WGAN(generator, discriminator, n_steps_in, n_steps_out, trial_id)
        Predicted_price, Real_price = wgan.train(X_train, y_train, past_y_train, epoch)

        # Rescale to original values
        rescaled_Real_price = y_scaler.inverse_transform(Real_price)
        rescaled_Predicted_price = y_scaler.inverse_transform(Predicted_price)

        # Predicted results and real value
        predict_result, real_price = real_predicted_results(rescaled_Predicted_price, rescaled_Real_price, index_train, output_dim)
    
        predict_result['predicted_mean'] = predict_result.mean(axis=1)
        real_price['real_mean'] = real_price.mean(axis=1)
    
        # Save train results for later use
        train_results = pd.concat([real_price['real_mean'],predict_result['predicted_mean']], axis=1)
        train_results.to_pickle(f'./outcome/predictions/Train_{trial_id}_{n_steps_in}_{n_steps_out}_{epoch-1}.pkl')

        # Plot the predicted result
        plt.figure(figsize=(16, 8))
        plt.plot(real_price["real_mean"])
        plt.plot(predict_result["predicted_mean"], color = 'r')
        plt.xlabel("Date")
        plt.ylabel("Stock price")
        plt.legend(("Real price", "Predicted price"), loc="upper left", fontsize=16)
        plt.title("Training set", fontsize=20)
        plt.savefig('./outcome/train_predictions.png')

        # Calculate RMSE, MAE, and MAPE
        predicted = predict_result["predicted_mean"]
        real = real_price["real_mean"]
        RMSE = np.sqrt(mean_squared_error(predicted, real))
        MAE = mean_absolute_error(predicted,real)
        MAPE = mean_absolute_percentage_error(predicted,real)

        print('Train RMSE: ', RMSE)
        print('Train MAE: ', float(MAE))
        print('Train MAPE: ', "{0:.5f}%".format(MAPE))

        # Test performance
        generator_model = tf.keras.models.load_model(f'./outcome/gen_models/{trial_id}_{n_steps_in}_{n_steps_out}_{epoch-1}.h5')
        test_predicted, test_RMSE, test_MAE, test_MAPE = get_test_plot(generator_model, X_test, y_test, y_scaler, index_test, output_dim, title='%s' % (stock))

        # Save predictions for later use
        output_dim = y_test.shape[1]
        y_predicted = generator_model(X_test)
        rescaled_real_y = y_scaler.inverse_transform(y_test)
        rescaled_predicted_y = y_scaler.inverse_transform(y_predicted)
        predict_result, real_price = real_predicted_results(rescaled_predicted_y, rescaled_real_y, index_test, output_dim)

        predict_result['predicted_mean'] = predict_result.mean(axis=1)
        real_price['real_mean'] = real_price.mean(axis=1)

        test_results = pd.concat([real_price['real_mean'],predict_result['predicted_mean']], axis=1)
        test_results.to_pickle(f'./outcome/predictions/Test_{trial_id}_{n_steps_in}_{n_steps_out}_{epoch-1}.pkl')