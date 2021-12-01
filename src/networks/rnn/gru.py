from math import exp
import torch


class GatedRecurrentUnit:
    """
    Code for the gated recurrent unit cell
    """
    def __init__(self, previous_hidden_size, input_size):
        self.previous_hidden_size = previous_hidden_size
        self.input_size = input_size

        # weights for update gate
        self.weight_input_update_gate = torch.rand((self.input_size, self.previous_hidden_size))
        self.weight_previous_hidden_state_update_gate = torch.rand(
            (self.previous_hidden_size, self.previous_hidden_size))

        # weights for reset gate
        self.weight_input_reset_gate = torch.rand((self.input_size, self.previous_hidden_size))
        self.weight_previous_hidden_state_reset_gate = torch.rand(
            (self.previous_hidden_size, self.previous_hidden_size))

        # weights for candidate activation
        self.weight_input_candidate_activation = torch.rand((self.input_size, self.previous_hidden_size))
        self.weight_previous_hidden_state_candidate_activation = torch.rand(
            (self.previous_hidden_size, self.previous_hidden_size))

        # bias for update gate, reset gate and candidate activation
        self.bias_update_gate = torch.rand(self.previous_hidden_size)
        self.bias_reset_gate = torch.rand(self.previous_hidden_size)
        self.bias_candidate_activation = torch.rand(self.previous_hidden_size)

    def forward_pass(self, previous_hidden_state, input_vector):
        """
        calculate the forward pass
        :param previous_hidden_state: vector of previous hidden state
        :param input_vector: input vector for word
        :return: the output from the GRU
        """
        # calculate the vector for update gate
        update_gate_vector = torch.sigmoid(torch.matmul(input_vector, self.weight_input_update_gate) +
                                           torch.matmul(previous_hidden_state,
                                                        self.weight_previous_hidden_state_update_gate) +
                                           self.bias_update_gate)

        # calculate the vector for reset gate
        reset_gate_vector = torch.sigmoid(torch.matmul(input_vector, self.weight_input_reset_gate) +
                                          torch.matmul(previous_hidden_state,
                                                       self.weight_previous_hidden_state_reset_gate) +
                                          self.bias_reset_gate)

        # calculate the vector for candidate activation
        candidate_activation = torch.tanh(torch.matmul(input_vector, self.weight_input_candidate_activation) +
                                          torch.matmul(self.previous_hidden_size * reset_gate_vector,
                                                       self.weight_previous_hidden_state_candidate_activation) +
                                          self.bias_candidate_activation)

        # calculate output vector
        output_vector = (1 - update_gate_vector) * previous_hidden_state + update_gate_vector * candidate_activation

        return output_vector
