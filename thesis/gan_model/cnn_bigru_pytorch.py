import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, feature_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=feature_size, out_channels=32, kernel_size=2, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1),
            # nn.Dropout(p=0.2),
            nn.GRU(input_size=64, hidden_size=64, batch_first=True, dropout=0.2, bidirectional=True),
            nn.Flatten(),
            nn.Linear(in_features=64, out_features=64, bias=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(in_features=64, out_features=32, bias=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(in_features=32, out_features=output_dim, bias=True)
        )
    
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, n_steps_in, n_steps_out):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=2, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Flatten(),
            nn.Linear(in_features=(n_steps_in + n_steps_out)*64, out_features=64, bias=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=64, out_features=32, bias=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=32, out_features=1, bias=True)
        )
    
    def forward(self, x):
        return self.model(x)

class WGAN(nn.Module):
    def __init__(self, generator, discriminator, n_steps_in, n_steps_out):
        super(WGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out
        self.batch_size = 32
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0001, betas=(0.5, 0.9))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0004, betas=(0.5, 0.9))

    def gradient_penalty(self, real_output, generated_output):
        """ Calculates the gradient penalty."""
        # get the interpolated data
        alpha = torch.randn((self.batch_size, self.n_steps_in + self.n_steps_out, 1), dtype=torch.float32)
        diff = generated_output - real_output
        interpolated = real_output + alpha * diff

        interpolated = torch.autograd.Variable(interpolated, requires_grad=True)
        # 1. Get the discriminator output for this interpolated data.
        pred = self.discriminator(interpolated)

        # 2. Calculate the gradients w.r.t to this interpolated data.
        grads = torch.autograd.grad(outputs=pred, inputs=interpolated, grad_outputs=torch.ones_like(pred),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

        # 3. Calculate the norm of the gradients
        norm = torch.sqrt(torch.sum(grads ** 2, dim=[1, 2]))

        gp = torch.mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        real_input, real_price, past_y = data
        self.batch_size = real_input.size(0)
        
        #Train the discriminator (5 times)
        for i in range(5):
            self.d_optimizer.zero_grad()
            # generate fake output
            generated_data = self.generator(real_input)
            # reshape the data
            generated_data_reshape = generated_data.view(generated_data.size(0), generated_data.size(1), 1)
            generated_output = torch.cat([generated_data_reshape, past_y], dim=1)
            real_y_reshape = real_price.view(real_price.size(0), real_price.size(1), 1)
            real_output = torch.cat([real_y_reshape, past_y], dim=1)
            # Get the logits for the real data
            D_real = self.discriminator(real_output)
            # Get the logits for the generated data
            D_generated = self.discriminator(generated_output)            
            # Calculate discriminator loss using generated and real logits
            real_loss = D_real.mean()
            generated_loss = D_generated.mean()
            d_cost = generated_loss - real_loss
            # Calculate the gradient penalty
            gp = self.gradient_penalty(real_output, generated_output)
            # Add the gradient penalty to the original discriminator loss
            d_loss = d_cost + gp * 10
            
            # Get the gradients w.r.t the discriminator loss
            d_loss.backward()
            # Update the weights of the discriminator
            self.d_optimizer.step()
        
        #Train the generator
        self.g_optimizer.zero_grad()
        generated_data = self.generator(real_input)
        generated_data_reshape = generated_data.view(generated_data.size(0), generated_data.size(1), 1)
        generated_output = torch.cat([generated_data_reshape, past_y], dim=1)
        real_y_reshape = real_price.view(real_price.size(0), real_price.size(1), 1)
        real_output = torch.cat([real_y_reshape, past_y], dim=1)
        D_generated = self.discriminator(generated_output)
        g_loss = -D_generated.mean()
        g_loss.backward()
        self.g_optimizer.step()

    def train(self, X_train, y_train, past_y, epochs):
        data = X_train, y_train, past_y
        train_hist = {}
        train_hist['D_losses'] = []
        train_hist['G_losses'] = []
        train_hist['per_epoch_times'] = []
        train_hist['total_ptime'] = []

        for epoch in range(epochs):
            start = time.time()

            real_price, generated_price, loss = self.train_step(data)

            G_losses = []
            D_losses = []

            Real_price = []
            Predicted_price = []

            D_losses.append(loss['discriminator_loss'].item())
            G_losses.append(loss['generator_loss'].item())

            Predicted_price.append(generated_price)
            Real_price.append(real_price)

            # Save the model every 100 epochs
            if (epoch + 1) % 100 == 0:
                    torch.save(self.generator.state_dict(), './outcome/gen_models/%s_%d_%d_%d.pth' % (self.trial_id, self.n_steps_in, self.n_steps_out, epoch))
                    print('epoch', epoch+1, 'discriminator_loss', loss['discriminator_loss'].item(), 'generator_loss', loss['generator_loss'].item())
            
            # For printing loss
            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - start
            train_hist['D_losses'].append(D_losses)
            train_hist['G_losses'].append(G_losses)
            train_hist['per_epoch_times'].append(per_epoch_ptime)

        # Reshape the predicted and real price
        Predicted_price = np.array(Predicted_price)
        Predicted_price = Predicted_price.reshape(Predicted_price.shape[1], Predicted_price.shape[2])
        Real_price = np.array(Real_price)
        Real_price = Real_price.reshape(Real_price.shape[1], Real_price.shape[2])

        # Plot the loss
        plt.figure(figsize=(16, 8))
        plt.plot(train_hist['D_losses'], label='D_loss')
        plt.plot(train_hist['G_losses'], label='G_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('./outcome/train_loss.png')
        #plt.show()

        return Predicted_price, Real_price
