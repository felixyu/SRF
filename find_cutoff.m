function cutoff = find_cutoff(dim, eps, sigma)
% Simple approximate inversion function to find value beyond which the 
% pdf has negligibly small mass.
    if dim < 200
        x = -2/(dim-1) * (eps .* sigma * gamma(dim/2)).^(2/(dim-1));
    else
        x = -2/(dim-1) * (eps .* sigma).^(2/(dim-1)) * (4*pi/dim)^(1/(dim-1)) * (exp(-1)*(dim/2 + 1/(6*dim - 1/5*dim)))^(1/(1-1/dim));
    end
    cutoff = sigma .* sqrt(-2*(dim-1)*Lambert_W(x,-1));
end