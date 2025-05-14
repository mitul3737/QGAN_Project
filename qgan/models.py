import torch
import pennylane as qml
from torch import nn
import torch.nn.functional as F

class DiscriminatorQuantumCircuit(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = qml.device("default.qubit", wires=self.n_qubits)
        
        # Create trainable parameters
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits))
        
        # Define the quantum circuit
        self.circuit = qml.QNode(self._circuit, self.device, interface="torch")

    def _circuit(self, inputs, weights):
        # Input encoding
        for wire in range(self.n_qubits):
            qml.RY(inputs[wire], wires=wire)
        
        # Variational layers
        for layer in range(self.n_layers):
            # Entanglement
            for wire in range(self.n_qubits):
                qml.CNOT(wires=[wire, (wire + 1) % self.n_qubits])
            # Rotation gates with trainable parameters
            for wire in range(self.n_qubits):
                qml.RZ(weights[layer, wire], wires=wire)
        
        # Return single expectation value
        return qml.expval(qml.PauliZ(0))

    def forward(self, inputs):
        # Process batch of inputs
        outputs = torch.zeros(inputs.shape[0], dtype=torch.float32)
        for i, input in enumerate(inputs):
            # Get quantum measurement (-1 to 1)
            quantum_output = self.circuit(input, self.weights)
            # Scale to (0,1) and apply sigmoid
            outputs[i] = torch.sigmoid((quantum_output + 1) / 2)
        return outputs

class GeneratorQuantumCircuit(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = qml.device("default.qubit", wires=self.n_qubits)
        
        # Create trainable parameters
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits))
        
        # Define the quantum circuit
        self.circuit = qml.QNode(self._circuit, self.device, interface="torch")

    def _circuit(self, noise, weights):
        # Noise encoding
        for wire in range(self.n_qubits):
            qml.RX(noise[wire], wires=wire)
        
        # Variational layers
        for layer in range(self.n_layers):
            # Entanglement
            for wire in range(self.n_qubits):
                qml.CNOT(wires=[wire, (wire + 1) % self.n_qubits])
            # Rotation gates with trainable parameters
            for wire in range(self.n_qubits):
                qml.RY(weights[layer, wire], wires=wire)
        
        # Return all qubit measurements
        return [qml.expval(qml.PauliZ(wire)) for wire in range(self.n_qubits)]

    def forward(self, noise):
        # Process batch of noise vectors
        outputs = torch.zeros((noise.shape[0], self.n_qubits), dtype=torch.float32)
        for i, n in enumerate(noise):
            # Scale quantum measurements from [-1,1] to [0,1]
            quantum_outputs = torch.tensor(self.circuit(n, self.weights))
            outputs[i] = (quantum_outputs + 1) / 2
        return outputs
