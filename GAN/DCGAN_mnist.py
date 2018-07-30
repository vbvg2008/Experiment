import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from tensorflow.keras.layers import Input,ELU,Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam

# The results are a little better when the dimensionality of the random vector is only 10.
# The dimensionality has been left at 100 for consistency with other GAN implementations.
randomDim = 100

# Load MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data('/home/jenno/Desktop/data/mnist/mnist.npz')
X_train = (X_train.astype(np.float32) - 127.5)/127.5

#reshape X_train from (n, 28, 28) to (n, 28, 28, 1)
X_train = np.expand_dims(X_train, -1)

# Optimizer
adam = Adam(lr = 0.0002, beta_1 = 0.5)

# Generator
generator = Sequential()

generator.add(Conv2DTranspose(filters=128, 
                              kernel_size= (7,7), 
                              input_shape=(1,1, randomDim)))
generator.add(BatchNormalization())
generator.add(ELU())

generator.add(Conv2DTranspose(filters=64, 
                              kernel_size= (5,5),
                              strides= (2,2),
                              padding= 'same'))
generator.add(BatchNormalization())
generator.add(ELU())

generator.add(Conv2DTranspose(filters=1, 
                              kernel_size= (5,5),
                              strides= (2,2),
                              activation='tanh',
                              padding= 'same'))

generator.compile(loss='binary_crossentropy', optimizer=adam)

# Discriminator
discriminator = Sequential()
discriminator.add(Conv2D(32, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=(28, 28, 1)))
discriminator.add(BatchNormalization())
discriminator.add(ELU())

discriminator.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same'))
discriminator.add(BatchNormalization())
discriminator.add(ELU())

discriminator.add(Conv2D(1, kernel_size=(7, 7), activation='sigmoid'))
discriminator.add(Flatten())
discriminator.compile(loss='binary_crossentropy', optimizer=adam)

# Combined network
discriminator.trainable = False
ganInput = Input(shape=(1,1, randomDim))
x = generator(ganInput)
ganOutput = discriminator(x)
gan = Model(inputs=ganInput, outputs=ganOutput)
gan.compile(loss='binary_crossentropy', optimizer=adam)

dLosses = []
gLosses = []

# Plot the loss from each batch
def plotLoss(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminitive loss')
    plt.plot(gLosses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('images/gan_loss_epoch_%d.png' % epoch)

# Create a wall of generated MNIST images
def plotGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, 1, 1, randomDim])
    generatedImages = generator.predict(noise)
    generatedImages = generatedImages.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('images/gan_generated_image_epoch_%d.png' % epoch)

# Save the generator and discriminator networks (and weights) for later use
def saveModels(epoch):
    generator.save('models/gan_generator_epoch_%d.h5' % epoch)
    discriminator.save('models/gan_discriminator_epoch_%d.h5' % epoch)
    
def train(epochs=1, batchSize=128):
    batchCount = X_train.shape[0] // batchSize
    for e in range(epochs):
        print('Epoch %d' % e)
        epoch_dLoss = []
        epoch_gLoss = []
        for i in range(batchCount):
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batchSize, 1, 1, randomDim])
            imageBatch = X_train[np.random.randint(0, X_train.shape[0], size=batchSize)]

            # Generate fake MNIST images
            generatedImages = generator.predict(noise)
            # print np.shape(imageBatch), np.shape(generatedImages)
            X = np.concatenate([imageBatch, generatedImages])

            # Labels for generated and real data
            yDis = np.zeros(2*batchSize)
            # One-sided label smoothing
            yDis[:batchSize] = 0.9

            # Train discriminator
            discriminator.trainable = True
            dloss = discriminator.train_on_batch(X, yDis)

            # Train generator
            noise = np.random.normal(0, 1, size=[batchSize, 1, 1, randomDim])
            yGen = np.ones(batchSize)
            discriminator.trainable = False
            gloss = gan.train_on_batch(noise, yGen)

            epoch_dLoss.append(dloss)
            epoch_gLoss.append(gloss)
        average_dloss = np.mean(epoch_dLoss)
        average_gloss = np.mean(epoch_gLoss)
        print('generator loss: ' + str(average_gloss))
        print('discriminator loss: ' + str(average_dloss))
        dLosses.append(average_dloss)
        gLosses.append(average_gloss)
        if e % 50 == 0 and e != 0:
            plotGeneratedImages(e)
            saveModels(e)

    # Plot losses from every epoch
    plotLoss(e)
    print('training finished')

if __name__ == '__main__':
    train(300, 128)
