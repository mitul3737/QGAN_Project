
# 🌟 Quantum Generative Adversarial Network (QGAN) 🌟
---

Welcome to the **Quantum Generative Adversarial Network (QGAN)**! This project explores the fusion of **quantum computing** and **generative adversarial networks (GANs)** to push the boundaries of machine learning and quantum technology.

## 🚀 What is QGAN?

**QGAN** combines the powerful framework of **GANs** with **quantum circuits** to generate data in quantum-enhanced ways. Leveraging **PennyLane** and **PyTorch**, this implementation enables quantum-inspired generators and discriminators, promising potential breakthroughs in various domains such as **quantum chemistry**, **finance**, and **data generation**.

## ✨ Key Features

- **Quantum Discriminator & Generator**: Powered by quantum circuits using the PennyLane library.
- **Customizable Quantum Layers**: Configure the number of qubits, quantum gates, and layers in the quantum circuits.
- **Fully Integrated Training Pipeline**: A complete pipeline to train the QGAN models with ease.
- **Quantum Chemistry Support**: Quantum simulations with molecular geometry support.
- **Pythonic**: Built with Python, using **PyTorch** for optimization and training.

## 📥 Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/QuantaScriptor/QGAN_Project.git
    cd QGAN_Project-main
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the deployment script to start training:
    ```bash
    bash deployment/deploy.sh
    ```

## 📚 Usage

After installing the dependencies, you can modify the `train.py` script or directly run it for training your QGAN model. The key commands:

- **Training the QGAN model**:
    ```bash
    python qgan/train.py
    ```
- **Interacting with molecular problems**: You can simulate molecules using the built-in `Molecule()` function to create molecular geometries.
    ```python
    from qgan.utils import Molecule
    molecule = Molecule("H2O")  # Simulates H2O molecule
    ```

## ⚛️ Quantum Computing Meets GANs

This project introduces a unique quantum twist on **GANs**:
- **DiscriminatorQuantumCircuit**: A quantum circuit acting as the discriminator, capable of detecting real vs. fake quantum states.
- **GeneratorQuantumCircuit**: A quantum circuit that generates synthetic data based on a latent quantum state.

By leveraging quantum gates, qubits, and quantum entanglement, the **QGAN** may offer faster convergence, better data generation, and possibilities for applications in **quantum chemistry**, **quantum finance**, and **data science**.

## 🧠 Concepts Behind QGAN

- **GANs (Generative Adversarial Networks)**: GANs are a class of machine learning frameworks where two neural networks contest with each other in a game. In QGAN, these are replaced by quantum circuits.
- **Quantum Circuits**: At the heart of QGAN, quantum circuits encode data and perform quantum operations to simulate and compute outputs in ways classical computers cannot.
- **Quantum Chemistry**: You can define molecular structures and perform simulations using `pennylane_qchem`.

## 🛠️ How It Works

1. **Generator** creates quantum states that resemble real data.
2. **Discriminator** evaluates if the generated data is real or fake.
3. The system iterates until the generator produces quantum data indistinguishable from real data.

## ⚡ System Requirements

- **Python 3.7+**
- **PennyLane** and **PyTorch**
- Optional: Quantum simulators (e.g., IBM Q, Rigetti) for running on real quantum devices

## 📜 License

This project is licensed under a **Commercial Use License and Public Non-Commercial License**. See the [LICENSE](./LICENSE) file for more details.

## 🙌 Acknowledgments

This project was developed by **Reece Colton Dixon** and utilizes the amazing tools from **PennyLane** and **PyTorch**.

---

🌐 **Quantum Generative Adversarial Networks (QGANs)** offer a glimpse into the future of **quantum machine learning**. Explore the world of quantum data generation today!
