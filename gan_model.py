import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose, LeakyReLU, Conv2D, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

def build_generator(latent_dim):
    model = Sequential()
    model.add(Dense(128 * 8 * 8, activation="relu", input_dim=latent_dim))
    model.add(Reshape((8, 8, 128)))
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(3, (7,7), activation='tanh', padding='same'))
    return model

def build_discriminator():
    input_shape = (None, None, 3)
    model = Sequential()
    model.add(Conv2D(32, (3,3), strides=(2,2), padding='same', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.25))
    
    model.add(GlobalAveragePooling2D())
    
    model.add(Dense(1, activation='sigmoid'))
    return model

def generate_and_save_images(generator, epoch, latent_dim, examples=100, dim=(10,10), figsize=(10,10)):
    noise = np.random.normal(0, 1, [examples, latent_dim])
    generated_images = generator.predict(noise)
    generated_images = (generated_images + 1) / 2.0

    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image_epoch_%d.png' % epoch)

def train_gan(epochs, batch_size, latent_dim, img_size=(32, 32)):
    (X_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()
    
    X_train = tf.image.resize(X_train, img_size)
    X_train = X_train / 127.5 - 1.0
    
    # Chuyển đổi X_train sang dạng numpy array
    X_train = X_train.numpy()

    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    generator = build_generator(latent_dim)
    discriminator.trainable = False
    gan = Sequential([generator, discriminator])
    gan.compile(loss='binary_crossentropy', optimizer=Adam())

    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, real)

        if (epoch + 1) % 1000 == 0:
            print(f"{epoch + 1} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")
            generate_and_save_images(generator, epoch + 1, latent_dim)

if __name__ == "__main__":
    epochs = 10000
    batch_size = 64
    latent_dim = 100
    train_gan(epochs, batch_size, latent_dim)
