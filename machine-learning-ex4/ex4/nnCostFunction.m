function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
size(X, 1)
size(X, 2)
size(Theta2, 1)
size(Theta2, 2)
size(Theta1, 1)
size(Theta1, 2)

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

tmpX = [ones(m, 1) X];
z_two = tmpX * Theta1';
a_two = sigmoid (z_two);
size(a_two, 1)
size(a_two, 2)
a_two = [ones(m, 1) a_two];
z_three = a_two * Theta2';
a_three = sigmoid(z_three);
size(a_three, 1)
size(a_three, 2)

yd = eye(num_labels);
y = yd(y, :);

logisf = (-y) .* log(a_three) - (1 - y) .* log(1 - a_three);
J = (1 / m) .* sum( sum ( logisf ) );

Theta1FromSecond = Theta1(:, 2:end);
Theta2FromSecond = Theta2(:, 2:end);
J = J + (lambda / (2 * m )) * ( sum ( sum ( Theta1FromSecond.^2 ) ) + sum ( sum ( Theta2FromSecond.^2 ) ) );


% back propagation algorithm implementation
delta_3 = a_three - y;
delta_2 = ( delta_3 * Theta2 ) (:, 2:end) .* sigmoidGradient( z_two );

triangle_2 = delta_3' * a_two; %10 * 5000 * 5000 * 26
triangle_1 = delta_2' * tmpX; %25 * 5000 * 5000 * 401

Theta1_grad = triangle_1 ./ m;
Theta2_grad = triangle_2 ./ m;

% back propagation algorithm regularized implementation
Theta1 = [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
Theta2 = [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];

Theta1_grad += (lambda / m) * Theta1;
Theta2_grad += (lambda / m) * Theta2;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end