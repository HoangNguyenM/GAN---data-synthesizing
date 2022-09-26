import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# prepare the data

df = pd.read_csv("D:/Dropbox/Code/GAN/Data/creditcard_train.csv")
col = df.columns

y = df['Class']
X = df.drop('Class',axis=1)

#%%

from tensorflow.python.keras.layers import Input, BatchNormalization
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers.core import Dense, Dropout
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras import backend
from tensorflow.python.keras.layers import Lambda

# build Semi-supervised GAN with a generator and a discriminator

randomDim = 29

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

generator = build_generator(latent_dim=randomDim, data_dim=29)
    
from tensorflow.keras.optimizers import Adam
opt = Adam(learning_rate=0.0002, beta_1=0.5)

def custom_activation(output):
    logexpsum = backend.sum(backend.exp(output),axis=-1,keepdims=True)
    result = logexpsum / (logexpsum + 1.0)
    return result

def build_discriminator(data_dim):
    model = Sequential()
    model.add(Dense(512,input_dim=data_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512,input_dim=data_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Dense(256,input_dim=data_dim))
    model.add(LeakyReLU(alpha=0.2))
    
    img = Input(shape=(data_dim,))
    features = model(img)
    
    # supervised model
    C_out = Dense(2, activation='softmax')(features)
    C_model = Model(img, C_out)
    C_model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=opt, metrics=['accuracy'])
    
    # unsupervised model
    D_out = Dense(1, activation=Lambda(custom_activation))(features)
    D_model = Model(img, D_out)
    D_model.compile(loss='binary_crossentropy', optimizer=opt)

    return D_model, C_model

D_model, C_model = build_discriminator(data_dim=29)

D_model.trainable = False

noise = Input(shape=(randomDim,))
gen_out = generator(noise)
gan_output = D_model(gen_out)
gan_model = Model(noise, gan_output)
gan_model.compile(loss='binary_crossentropy',optimizer='adam')

#%%

# define a train function using Gradient descent and train both the generator and the discriminator

from imblearn.under_sampling import RandomUnderSampler

def train(X_train, y_train, epochs, batch_size=128):
    half_batch = int(batch_size/2)
    
    for epoch in range(epochs):
    
        rus = RandomUnderSampler(sampling_strategy='not minority')
        X_res, y_res = rus.fit_resample(X_train, y_train)
        X_res = X_res.values
        
        # select a random half batch of real data with labels
        idx = np.random.randint(0, X_res.shape[0], half_batch)
        real_data = X_res[idx]
        real_labels = y_res[idx]
        
        # Sample noise and generate half batch of fake data
        noise = np.random.normal(0,1, (half_batch, randomDim))
        gen_data = generator.predict(noise)
        
        valid = np.ones((half_batch, 1))
        fake = np.zeros((half_batch, 1))
        
        # Train the discriminator
        C_loss = C_model.train_on_batch(real_data, real_labels)
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

train(X, y, epochs=700, batch_size=128)

#%%

# we use PCA plot to observe the difference between the real data and synthetic data

from sklearn.decomposition import PCA

noise = np.random.normal(0,1,(500,randomDim))
gen_data = generator.predict(noise)

rus = RandomUnderSampler(sampling_strategy='not minority')
X_res, y_res = rus.fit_resample(X, y)
X_res = X_res.values

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
