
# Quantum Generative Adversarial Network (QGAN)
# Developed by Reece Colton Dixon
# License: Commercial Use License and Public Non-Commercial License
# For license information, see the LICENSE file

import torch
import pennylane as qml
import pennylane_qchem as qchem
from qgan.models import DiscriminatorQuantumCircuit, GeneratorQuantumCircuit

# Binary Cross-Entropy Loss function
criterion = torch.nn.BCELoss()

# Define Hyperparameters
epochs = 100
batch_size = 64
learning_rate = 0.0002

def qgan_loss(generated_samples, real_samples, discriminator):
    discriminator_real_output = discriminator(real_samples)
    discriminator_fake_output = discriminator(generated_samples)

    generator_loss = -torch.mean(discriminator_fake_output)
    discriminator_loss = criterion(discriminator_real_output, torch.ones_like(discriminator_real_output)) +                          criterion(discriminator_fake_output, torch.zeros_like(discriminator_fake_output))

    return generator_loss, discriminator_loss

def main():
    discriminator = DiscriminatorQuantumCircuit()
    generator = GeneratorQuantumCircuit()
    optimizer_gen = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

    # Assuming we have a data loader function `get_data_loader()` returning batches of real_samples
    for epoch in range(epochs):
        for real_samples in get_data_loader(batch_size):
            # Generate fake samples from the generator
            latent_space_samples = torch.randn(batch_size, 100)  # Latent space for GAN
            generated_samples = generator(latent_space_samples)
            
            # Compute losses
            gen_loss, disc_loss = qgan_loss(generated_samples, real_samples, discriminator)

            # Update Generator
            optimizer_gen.zero_grad()
            gen_loss.backward()
            optimizer_gen.step()

            # Update Discriminator
            optimizer_disc.zero_grad()
            disc_loss.backward()
            optimizer_disc.step()

        print(f'Epoch [{epoch + 1}/{epochs}] | Generator Loss: {gen_loss.item()} | Discriminator Loss: {disc_loss.item()}')

if __name__ == "__main__":
    main()
