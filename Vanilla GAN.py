import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# prepare the data

df = pd.read_csv("D:/Dropbox/Code/GAN/Data/heart_train.csv")

X = df.drop('DEATH',axis=1)
col = X.columns
X = X.to_numpy()
print(X.shape)

#%%

from tensorflow.python.keras.layers import Input, BatchNormalization
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers.core import Dense, Dropout
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU

# build Vanilla GAN with a generator and a discriminator

randomDim = 12

def build_generator(latent_dim,data_dim):
    model = Sequential()
    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    
    model.add(Dense(data_dim,activation='tanh'))
    
    noise = Input(shape=(latent_dim,))
    img = model(noise)

    return Model(noise, img)

generator = build_generator(latent_dim=randomDim, data_dim=12)
    
from tensorflow.keras.optimizers import Adam
opt = Adam(learning_rate=0.0002, beta_1=0.5)

def build_discriminator(data_dim):
    model = Sequential()
    model.add(Dense(512,input_dim=data_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512,input_dim=data_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Dense(256,input_dim=data_dim))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dense(1, activation='sigmoid'))
    img = Input(shape=(data_dim,))
    validity = model(img)

    return Model(img, validity)

D_model = build_discriminator(data_dim=12)
D_model.compile(loss='binary_crossentropy', optimizer=opt)


D_model.trainable = False

noise = Input(shape=(randomDim,))
gen_out = generator(noise)
gan_output = D_model(gen_out)
gan_model = Model(noise, gan_output)
gan_model.compile(loss='binary_crossentropy',optimizer=opt)

#%%

# define train function using Gradient descent

def train(X_train, epochs, batch_size=128):
    half_batch = int(batch_size/2)
    
    for epoch in range(epochs):

        # select a random half batch of real data with labels
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        real_data = X_train[idx]
        
        # Sample noise and generate half batch of fake data
        noise = np.random.normal(0,1, (half_batch, randomDim))
        gen_data = generator.predict(noise)
        
        valid = np.ones((half_batch, 1))
        fake = np.zeros((half_batch, 1))
        
        # Train the discriminator

        D_loss_real = D_model.train_on_batch(real_data, valid)
        D_loss_fake = D_model.train_on_batch(gen_data, fake)
        D_loss = 0.5 * np.add(D_loss_real, D_loss_fake)
        
        # Train the generator
        noise = np.random.normal(0,1, (batch_size, randomDim))
        validity = np.ones((batch_size, 1))
        
        G_loss = gan_model.train_on_batch(noise, validity)
        
        print("%d : %.2f%%" %(epoch, 100*D_loss))

    
#%%
# train GAN

train(X, epochs=1000, batch_size=128)

#%%

# we use PCA plot to observe the difference between the real data and synthetic data

from sklearn.decomposition import PCA

noise = np.random.normal(0,1,(300,randomDim))
gen_data = generator.predict(noise)

gen_data_finished = pd.DataFrame()
for i in range(gen_data.shape[1]):
    gen_data_finished[col[i]] = gen_data[:,i]    

print(gen_data_finished)

categorical = ['anaemia','diabetes','high_blood_pressure','sex','smoking']
for col in categorical:
    for i in range(gen_data_finished.shape[0]):
        if gen_data_finished[col].iloc[i] > 0:
            gen_data_finished[col].iloc[i] = 1
        else:
            gen_data_finished[col].iloc[i] = -1
            
print(gen_data_finished)

# PCA
prep_data = X
prep_data_hat = gen_data_finished

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
