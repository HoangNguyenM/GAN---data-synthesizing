from ctgan.synthesizers.ctgan import CTGANSynthesizer
import pandas as pd
import matplotlib.pyplot as plt

# prepare the data

X = pd.read_csv("D:/Dropbox/Code/GAN/Data/heart_train.csv")
col = X.columns
print(col)

categorical = ['anaemia','diabetes','high_blood_pressure','sex','smoking']
X = X.drop('DEATH',axis=1)
print(X.shape)

# import CTGAN from the library and train the model

model = CTGANSynthesizer()
model.fit(X,categorical)

#%%

# we use PCA plot to observe the difference between the real data and synthetic data

from sklearn.decomposition import PCA

gen_data = model.sample(300)

# PCA
prep_data = X
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