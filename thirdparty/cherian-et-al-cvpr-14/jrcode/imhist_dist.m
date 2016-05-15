function dist = imhist_dist(im1, im2, bins_per_channel)
%IMHIST_DIST Compute the cosine similarity between the histograms
%associated with two images.
im1_hist = get_rgb_hist(im1, bins_per_channel);
im2_hist = get_rgb_hist(im2, bins_per_channel);
dist = acos(dot(im1_hist, im2_hist) / (norm(im1_hist) * norm(im2_hist)));
end

function rv = get_rgb_hist(im, bins_per_channel)
rv = zeros([bins_per_channel * size(im, 3), 1]);
for chan=1:size(im, 3)
    [rv((chan-1)*bins_per_channel+1:chan*bins_per_channel), ~] ...
        = imhist(im(:, :, chan), bins_per_channel);
end
end