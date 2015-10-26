function rgb_mat = visualise_flow(flow_mat)
%VISUALISE_FLOW Make flow look pretty.

flow_size = size(flow_mat);
flat_flow = reshape(flow_mat, [flow_size(1) * flow_size(2), 2]);
hsv_vals = ones([flow_size(1) * flow_size(2), 3]);

% Angle of flow is hue (0 = 0 deg, 360 = 360 deg)
hsv_vals(:, 1) = atan2(flat_flow(:, 1), flat_flow(:, 2)) / (2 * pi) + 0.5;

% (1 - normalised magnitude of flow) is saturation.
mags = sum(flat_flow.^2, 2);
norm_mags = mags / max(mags(:));
hsv_vals(:, 2) = 1 - norm_mags;

% Now make the RGB image
out_size = [flow_size(1:2), 3];
rgb_mat = reshape(hsv2rgb(hsv_vals), out_size);

% Now do some funky quiver stuff (this is part of the same function because
% YOLO)
scale_factor = 0.05;
small_flow = imresize(flow_mat, scale_factor);
x_vals = linspace(1, size(flow_mat, 2), size(small_flow, 2));
y_vals = linspace(1, size(flow_mat, 1), size(small_flow, 1));
quiver(x_vals, y_vals, small_flow(:, :, 1), small_flow(:, :, 2), 0);
axis equal;
axis tight;
axis off;
xlim([1 size(flow_mat, 2)]);
ylim([1 size(flow_mat, 1)]);
end

