function [J] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
% Note: This version is modified to work with Grey Wolf Optimization.
% It returns only the cost J without the gradient.

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

J = 0;

% Forward propagation
y_temp = eye(num_labels);

% Input layer to hidden layer
X_with_bias = [ones(size(X, 1), 1) X]; % Add bias unit
z2 = X_with_bias * Theta1';
a2 = sigmoid(z2);

% Hidden layer to output layer
a2_with_bias = [ones(size(a2, 1), 1) a2]; % Add bias unit
z3 = a2_with_bias * Theta2';
h = sigmoid(z3);

% Prepare one-hot encoded y values
Y = zeros(m, num_labels);
for i = 1:m
  Y(i,:) = y_temp(y(i), :);
end

% Calculate cost without regularization
J = sum(sum((-Y) .* log(h) - (1 - Y) .* log(1 - h))) / m;

% Add regularization term
% Use columns() function since it's available in Octave
J = J + (lambda/(2*m)) * (sum(sum(Theta1(:, 2:columns(Theta1)).^2)) + ...
                           sum(sum(Theta2(:, 2:columns(Theta2)).^2)));

end
