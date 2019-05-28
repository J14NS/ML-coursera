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

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

Theta1_reduce = Theta1(:, 2:end);
Theta2_reduce = Theta2(:, 2:end);

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


for i = 1 : m;
	% Select one sample
	X_sample = X(i, :)';
	y_sample = zeros(num_labels, 1);
		y_label = y(i, 1);
		y_sample(y_label, 1) = 1;
		
	% Compute g value for hidden layer
	X_sample = [1; X_sample];
	g_value_hidden = sigmoid(Theta1 * X_sample);

	% Compute g value for output layer
	g_value_hidden = [1; g_value_hidden];
	g_value_output = sigmoid(Theta2 * g_value_hidden);
	
	% Compute the cost function
	J = J + ( - y_sample' * log(g_value_output) -  (ones(num_labels, 1) - y_sample)' * log(ones(num_labels, 1) - g_value_output) )/m;
	
	delta_3 = g_value_output - y_sample;
	Theta2_grad = Theta2_grad + delta_3 * g_value_hidden';
	
	g_grad = sigmoidGradient(Theta1 * X_sample);
	g_grad = [1; g_grad];
	delta_2 = Theta2' * delta_3 .* g_grad;
	delta_2 = delta_2(2:end);
	Theta1_grad = Theta1_grad + delta_2 * X_sample';
	
end

J = J + lambda / (2 * m) * (sum(sum(Theta1_reduce.^2)) + sum(sum(Theta2_reduce.^2))) ; 


Theta2_grad_avg = Theta2_grad / m + lambda * Theta2 / m ;
Theta1_grad_avg = Theta1_grad / m + lambda * Theta1 / m ;

Theta2_grad_avg(:, 1) = Theta2_grad(:, 1) / m;
Theta1_grad_avg(:, 1) = Theta1_grad(:, 1) / m;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients

grad = [Theta1_grad_avg(:); Theta2_grad_avg(:)];

end
