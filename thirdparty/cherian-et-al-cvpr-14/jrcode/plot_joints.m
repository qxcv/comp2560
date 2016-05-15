function plot_joints(locs, colour)
% Plot an N * 2 array of joints
if nargin < 2
    chosen_col = randi([1 7]);
    l = lines;
    colour = l(chosen_col, :);
end

extra_args = {'MarkerEdgeColor', colour, 'MarkerFaceColor', colour};

for i=1:size(locs, 1);
    x = locs(i, 1);
    y = locs(i, 2);
    h = plot(x, y, 'LineStyle', 'none', 'Marker', '+', 'MarkerSize', 15, extra_args{:});
    text(double(x+8), double(y+8), cellstr(num2str(i)), 'Color', h.MarkerEdgeColor, 'FontSize', 15);
end
end

