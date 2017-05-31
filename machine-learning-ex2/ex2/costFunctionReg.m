function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = size(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(n);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X*theta);
for i = 1:m
    J +=  -y(i)*log(h(i)) - (1-y(i))*log(1-h(i));
end

for j = 1:n
    for i = 1:m
        grad(j) += (h(i) - y(i)) * X(i,j);
	end
	if j > 1
        J += lambda * power(theta(j), 2) / 2;
        grad(j) += lambda * theta(j);
	end
end

J /= m;
grad /= m;
% =============================================================

end
