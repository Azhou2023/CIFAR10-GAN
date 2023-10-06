import numpy as np 
import tensorflow as tf 
import matplotlib as plt
from tensorflow import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten, LeakyReLU, Dropout, Reshape, Conv2DTranspose
from keras.optimizers import Adam
from keras.datasets.cifar10 import load_data


def build_discriminator():
    
    input = Input(shape=(32,32,3))

    c1 = Conv2D(64, (3,3), padding='same')(input)
    l1 = LeakyReLU(alpha=0.2)(c1)
    
    c2 = Conv2D(128, (3,3), strides=(2,2))(l1)
    l2 = LeakyReLU(alpha=0.2)(c2)

    c3 = Conv2D(128, (3,3), strides=(2,2))(l2)
    l3 = LeakyReLU(alpha=0.2)(c3)

    c4 = Conv2D(256, (3,3), strides=(2,2))(l3)
    l4 = LeakyReLU(alpha=0.2)(c4)
    
    f1 = Flatten()(l4)
    d1 = Dropout(0.4)(f1)
    output = Dense(1, activation="sigmoid")(d1)
    
    model = Model(inputs=[input], output=[output])

    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    
    return model 

def build_generator(latent_dim):
    

    input = Dense(4096, input_dim=latent_dim)
    l1 = LeakyReLU(alpha=0.2)(input)
    r1 = Reshape((4,4,256))(l1)
    
    c1 = Conv2DTranspose(128,(4,4), strides=(2,2), padding="same")(r1)
    l2 = LeakyReLU(alpha=0.2)(c1)
    
    c2 = Conv2DTranspose(128,(4,4), strides=(2,2), padding="same")(l2)
    l3 = LeakyReLU(alpha=0.2)(c2)

    c3 = Conv2DTranspose(128,(4,4), strides=(2,2), padding="same")(l3)
    l4 = LeakyReLU(alpha=0.2)(c2)
    
    c3 = Conv2D(3, (8,8), activation="tanh", padding="same")(l3)

    model = Model(inputs=[input], output=[c3])

    return model   
    
def build_GAN(generator, discriminator):
    discriminator.trainable = False
    
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt)
    
    return model

def generate_real_images(dataset, n_samples):
    rand = np.random.randint(0, dataset.shape[0], n_samples)
    X = dataset[rand]
    Y = tf.ones((n_samples, 1))
    return X, Y

def generate_latent_points(latent_dim, n_samples):
    x = np.random.randn(latent_dim * n_samples)
    x = x.reshape(n_samples, latent_dim)
    return x

def generate_fake_images(generator, n_samples, latent_dim):
    input = generate_latent_points(latent_dim, n_samples)
    X = generator.predict(input, verbose=None)
    Y = tf.zeros((n_samples, 1))
    return X, Y

def train(generator, discriminator, GAN, latent_dim, dataset):
    n_epochs = 100
    samples_per_batch = 128
    n_batches = int(dataset.shape[0]/samples_per_batch)
    half_batch = 64
    for i in range(n_epochs):
        print('\n Epoch {}/{}'.format(i, n_epochs))
        progbar = tf.keras.utils.Progbar(n_batches)
        loss = 0
        for j in range(n_batches):
            x_real, y_real = generate_real_images(dataset, half_batch)
            discriminator.train_on_batch(x_real, y_real)
            
            x_fake, y_fake = generate_fake_images(generator, half_batch, latent_dim)
            discriminator.train_on_batch(x_fake, y_fake)
            
            latent_points = generate_latent_points(latent_dim, samples_per_batch)
            y = tf.ones((samples_per_batch, 1))
            
            loss = GAN.train_on_batch(latent_points, y)
            progbar.update(j+1)
        print(loss)
    generator.save("generator")
    
(trainX, _), (_, _) = load_data()
X = trainX.astype('float32')
X = (X-127.5)/127.5    

# dataset = tf.data.Dataset.from_tensor_slices(X)
# dataset = dataset.batch(128)

latent_dim = 100
generator = build_generator(latent_dim)
discriminator = build_discriminator()
GAN = build_GAN(generator, discriminator)

train(generator, discriminator, GAN, latent_dim, X)



    

            