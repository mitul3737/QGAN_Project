
# Quantum Generative Adversarial Network (QGAN)
# Developed by Reece Colton Dixon
# License: Commercial Use License and Public Non-Commercial License
# For license information, see the LICENSE file

import torch
import pennylane as qml

class DiscriminatorQuantumCircuit:
    def __init__(self, n_qubits=4, n_layers=2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(self.device, interface="torch")
        def circuit(self, inputs):
            for wire in range(self.n_qubits):
                qml.RY(inputs[wire], wires=wire)
            for _ in range(self.n_layers):
                for wire in range(self.n_qubits):
                    qml.CNOT(wires=[wire, (wire + 1) % self.n_qubits])
                    qml.RZ(0.1, wires=wire)
            return [qml.expval(qml.PauliZ(wire)) for wire in range(self.n_qubits)]

    def forward(self, inputs):
        return torch.tensor(self.circuit(inputs), requires_grad=True)

class GeneratorQuantumCircuit:
    def __init__(self, n_qubits=4, n_layers=2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(self.device, interface="torch")
        def circuit(self, inputs):
            for wire in range(self.n_qubits):
                qml.RX(inputs[wire], wires=wire)
            for _ in range(self.n_layers):
                for wire in range(self.n_qubits):
                    qml.CNOT(wires=[wire, (wire + 1) % self.n_qubits])
                    qml.RY(0.2, wires=wire)
            return [qml.expval(qml.PauliZ(wire)) for wire in range(self.n_qubits)]

    def forward(self, inputs):
        return torch.tensor(self.circuit(inputs), requires_grad=True)
