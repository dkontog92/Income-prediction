function cost  = LogComputeCost(X,y,theta, reg_coeff)

if nargin == 3
    reg_coeff = 0;
end

m = size(X,1);
hypothesis = sigmoid(X*theta);
cost = (1/m)*sum(- y.*log(hypothesis)-(1-y).*log(1-hypothesis) + (reg_coeff/(2*m))*sum(theta(2:end).^2));

end
