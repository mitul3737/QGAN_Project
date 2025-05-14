import torch
from torch.utils.data import DataLoader, TensorDataset
from models import GeneratorQuantumCircuit, DiscriminatorQuantumCircuit

def get_data_loader(batch_size):
    # Create quantum data in [0,1] range
    num_samples = 1000
    num_qubits = 4
    real_data = torch.rand(num_samples, num_qubits)  # Replace with actual quantum data
    dataset = TensorDataset(real_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def main():
    # Hyperparameters
    n_qubits = 4
    n_layers = 2
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.01
    
    # Initialize models
    generator = GeneratorQuantumCircuit(n_qubits, n_layers)
    discriminator = DiscriminatorQuantumCircuit(n_qubits, n_layers)
    
    # Optimizers
    optimizer_gen = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
    
    # Loss function
    loss_fn = torch.nn.BCELoss()
    
    # Training loop
    for epoch in range(num_epochs):
        for real_samples, in get_data_loader(batch_size):
            
            # 1. Train Discriminator on real samples
            optimizer_dis.zero_grad()
            real_predictions = discriminator(real_samples)
            real_targets = torch.ones_like(real_predictions)
            loss_real = loss_fn(real_predictions, real_targets)
            
            # 2. Generate fake samples
            noise = torch.rand(batch_size, n_qubits)  # [0,1] range
            fake_samples = generator(noise)
            
            # 3. Train Discriminator on fake samples
            fake_predictions = discriminator(fake_samples.detach())
            fake_targets = torch.zeros_like(fake_predictions)
            loss_fake = loss_fn(fake_predictions, fake_targets)
            
            # 4. Update Discriminator
            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimizer_dis.step()
            
            # 5. Train Generator
            optimizer_gen.zero_grad()
            gen_predictions = discriminator(fake_samples)
            gen_targets = torch.ones_like(gen_predictions)
            loss_g = loss_fn(gen_predictions, gen_targets)
            loss_g.backward()
            optimizer_gen.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, D Loss: {loss_d.item():.4f}, G Loss: {loss_g.item():.4f}")

if __name__ == "__main__":
    main()
