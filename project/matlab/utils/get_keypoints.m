%% get keypoints
function k = get_keypoints(b)
x1 = b(:, 1:4:end);
y1 = b(:, 2:4:end);
x2 = b(:, 3:4:end);
y2 = b(:, 4:4:end);
x = (x1+x2)/2; y = (y1+y2)/2;
k = interleave(x,y);
end%%
