function mu = mean_dists(j1, j2)
%MEAN_DISTS Compute mean Euclidean distance between sets of joints
assert(ismatrix(j1) && ismatrix(j2));
assert(size(j1, 2) == 2);
mu = mean(sqrt(sum((j1 - j2).^2, 2)));
assert(isscalar(mu));
end