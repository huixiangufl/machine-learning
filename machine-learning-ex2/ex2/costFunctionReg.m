function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

j = 2;
n = columns(X);
lambda_value = 0;
while j <= n;
	lambda_value = lambda_value + (lambda / (2 * m)) * (theta(j) ^ 2);
	j = j + 1;
end;

i = 1;
while i <= m;
	h_theta = sigmoid( X(i, :) * theta );
	tmp = y(i) * log(h_theta) + (1 - y(i)) * log(1 - h_theta);
	J = J + (-1 / m) * tmp;
	tmpVector = ( (1 / m) * (h_theta - y(i)) ) .* X(i, :);
	grad = grad + tmpVector';
	i = i + 1;
end;

J = J + lambda_value;
tmpTheta = theta;
tmpTheta(1) = 0;
grad = grad + (lambda / m) * tmpTheta;





% =============================================================

end
