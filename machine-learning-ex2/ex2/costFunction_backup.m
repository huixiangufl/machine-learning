function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t the parameters.

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
%
% Note: grad should have the same dimensions as theta
%
i = 1;
while i <= m;
	h_theta = sigmoid( X(i, :) * theta );
	tmp = y(i) * log(h_theta) + (1 - y(i)) * log(1 - h_theta);
	J = J + (-1 / m) * tmp;
	i = i + 1;
end;


%j = 1;
%while j <= rows(theta);
%	i = 1;
%	while i <= m;
%		h_theta = sigmoid( X(i, :) * theta);
%		grad(j) = grad(j) + (1 / m) * (h_theta - y(i)) * X(i, j);
%		i = i + 1;
%	end;
%	j = j + 1;
%end;

i = 1;
while i <= m;
	h_theta = sigmoid( X(i, :) * theta);
	tmpVector = ( (1 / m) * (h_theta - y(i)) ) .* X(i, :);
	grad = grad + tmpVector';
	i = i + 1;
end;





% =============================================================

end
