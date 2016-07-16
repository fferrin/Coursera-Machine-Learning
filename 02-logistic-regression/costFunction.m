function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0; % cost function
grad = zeros(size(theta)); % gradient

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% logistic regression hypothesis
h_theta = sigmoid(X * theta);
one = ones(m, 1);

% find indices of positive and negative examples
pos = find(y == 1);
neg = find(y == 0);

% compute cost function and gradient in logistic regression
% J(\theta) = \frac{1}{m} \sum_{i=1}^{m} {-y^{(i)} log(h_{\theta}(x^{(i)})) -
%              (1 - y^{(i)}) log(1 - h_{\theta}(x^{(i)}))}
J = -(y' * log(h_theta) + (one - y)' * log(one - h_theta)) / m;

% \frac{\partial J(\theta)}{\partial \theta_{j}} = \frac{1}{m} \sum_{i=1}^{m} {
% (h_{\theta}(x^{(i)}) - y^{(i)}) x_{j}^{(i)} }
grad = (h_theta - y)' * X / m;

% =============================================================

end
