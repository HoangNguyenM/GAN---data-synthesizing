import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler

from keras.layers import Input, BatchNormalization, Dense, Dropout, LeakyReLU, Lambda
from keras.models import Model, Sequential
from keras import backend

# prepare the data

df = pd.read_csv("data/creditcard_train.csv")
col = df.columns

y = df['Class']
X = df.drop('Class', axis=1)
X = X.to_numpy()
print(X.shape)

# build Semi-supervised GAN with a generator and a discriminator

randomDim = 29

def build_generator(latent_dim, data_dim):
    model = Sequential()
    model.add(Dense(256, input_dim = latent_dim))
    model.add(LeakyReLU(alpha = 0.2))
    model.add(BatchNormalization(momentum = 0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha = 0.2))
    model.add(BatchNormalization(momentum = 0.8))
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha = 0.2))
    model.add(BatchNormalization(momentum = 0.8))
    
    model.add(Dense(data_dim, activation='tanh'))
    
    noise = Input(shape = (latent_dim,))
    x = model(noise)

    return Model(noise, x)

generator = build_generator(latent_dim = randomDim, data_dim = 29)

def custom_activation(output):
    logexpsum = backend.sum(backend.exp(output), axis=-1, keepdims=True)
    result = logexpsum / (logexpsum + 1.0)
    return result

def build_discriminator(data_dim):
    model = Sequential()
    model.add(Dense(512, input_dim = data_dim))
    model.add(LeakyReLU(alpha = 0.2))
    model.add(Dense(256, input_dim = data_dim))
    model.add(LeakyReLU(alpha = 0.2))
    model.add(Dropout(0.25))
    model.add(Dense(128, input_dim = data_dim))
    model.add(LeakyReLU(alpha = 0.2))
    
    x = Input(shape = (data_dim,))
    features = model(x)
    
    # supervised model
    C_out = Dense(2, activation='softmax')(features)
    C_model = Model(x, C_out)
    C_model.compile(loss='sparse_categorical_crossentropy',
                    optimizer='adam', metrics=['accuracy'])
    
    # unsupervised model
    D_out = Dense(1, activation=Lambda(custom_activation))(features)
    D_model = Model(x, D_out)
    D_model.compile(loss='binary_crossentropy', optimizer='adam')

    return D_model, C_model

D_model, C_model = build_discriminator(data_dim=29)

D_model.trainable = False

noise = Input(shape=(randomDim,))
gen_out = generator(noise)
gan_output = D_model(gen_out)
gan_model = Model(noise, gan_output)
gan_model.compile(loss='binary_crossentropy', optimizer='adam')

# define a train function using Gradient descent and train both the generator and the discriminator

def train(X_train, y_train, epochs, batch_size = 32):
    half_batch = int(batch_size / 2)
    
    for epoch in range(epochs):
    
        rus = RandomUnderSampler(sampling_strategy = 'not minority')
        X_res, y_res = rus.fit_resample(X_train, y_train)
        
        # select a random half batch of real data with labels
        idx = np.random.randint(0, X_res.shape[0], half_batch)
        real_data = X_res[idx]
        real_labels = y_res[idx]
        
        # Sample noise and generate half batch of fake data
        noise = np.random.normal(0,1, (half_batch, randomDim))
        gen_data = generator.predict(noise, batch_size = half_batch, verbose = 0)
        
        valid = np.ones((half_batch, 1))
        fake = np.zeros((half_batch, 1))
        
        # Train the discriminator
        C_loss = C_model.fit(real_data, real_labels, batch_size = half_batch, verbose = 0)
        D_loss_real = D_model.fit(real_data, valid, batch_size = half_batch, verbose = 0)
        D_loss_fake = D_model.fit(gen_data, fake, batch_size = half_batch, verbose = 0)
        D_loss = 0.5 * np.add(D_loss_real.history['loss'][-1], D_loss_fake.history['loss'][-1])
        
        # Train the generator
        noise = np.random.normal(0,1, (batch_size, randomDim))
        validity = np.ones((batch_size, 1))
        
        G_loss = gan_model.fit(noise, validity, batch_size = half_batch, verbose = 0)
        
        # print out the loss
        if (epoch + 1) % 100 == 0:
            print("epoch %d: d loss %.2f, g loss %.2f" %(epoch + 1, D_loss, G_loss.history['loss'][-1]))
    
    
# train GAN

train(X, y, epochs = 700, batch_size = 128)

# we use PCA plot to observe the difference between the real data and synthetic data

noise = np.random.normal(0, 1, (500, randomDim))
gen_data = generator.predict(noise, verbose = 0)

rus = RandomUnderSampler(sampling_strategy = 'not minority')
X_res, y_res = rus.fit_resample(X, y)

# PCA
prep_data = X_res
prep_data_hat = gen_data

pca = PCA(n_components = 2)
pca.fit(prep_data)
pca_results = pca.transform(prep_data)
pca_hat_results = pca.transform(prep_data_hat)

# Plot
f, ax = plt.subplots(1)
plt.scatter(pca_results[:,0], pca_results[:,1],
            c = 'red', alpha = 0.2, label = "Original")
plt.scatter(pca_hat_results[:,0], pca_hat_results[:,1],
            c = 'blue', alpha = 0.2, label = "Synthetic")
ax.legend()
plt.show()
