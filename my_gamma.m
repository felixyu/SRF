function y = my_gamma(z) 
% Gamma function with simple approximation to prevent overflow.

    if z > 100
        y = sqrt(2*pi./z) .* (exp(-1).*(z + 1./(12*z-1/10./z))).^z;
    else
        y = gamma(z);
    end
end