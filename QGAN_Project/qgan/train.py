# Quantum Generative Adversarial Network (QGAN)
# Developed by Reece Colton Dixon
# License: Commercial Use License and Public Non-Commercial License
# For license information, see the LICENSE file

import torch
import pennylane as qml
import pennylane_qchem as qchem
from qgan.models import DiscriminatorQuantumCircuit, GeneratorQuantumCircuit

def qgan_loss(generated_samples, real_samples):
    discriminator_real_output = discriminator(real_samples)
    discriminator_fake_output = discriminator(generated_samples)

    generator_loss = -torch.mean(discriminator_fake_output)
    discriminator_loss = criterion(discriminator_real_output, torch.ones_like(discriminator_real_output)) +                          criterion(-discriminator_fake_output, torch.zeros_like(discriminator_fake_output))

    return generator_loss, discriminator_loss

def main():
    discriminator = DiscriminatorQuantumCircuit()
    generator = GeneratorQuantumCircuit()
    optimizer_gen = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
    num_epochs = 500

    for epoch in range(num_epochs):
        problem = interact_with_user_for_problem_definition()
        molecule = Molecule(problem['geometry'])
        hamiltonian, qubits = qchem.molecular_hamiltonian(molecule)
        noise = torch.randn(batch_size, n_qubits)
        generated_states = generator(noise)
        generator_loss, discriminator_loss = qgan_loss(generated_states, hamiltonian)

        optimizer_gen.zero_grad()
        generator_loss.backward()
        optimizer_gen.step()

        optimizer_disc.zero_grad()
        discriminator_loss.backward()
        optimizer_disc.step()

        print('Epoch [{}/{}], Generator Loss: {:.4f}, Discriminator Loss: {:.4f}'.format(epoch+1, num_epochs, generator_loss.item(), discriminator_loss.item()))

if __name__ == "__main__":
    main()
