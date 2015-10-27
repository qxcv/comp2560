function visualise_map(unaries)
%VISUALISE_MAP Draws a nice heatmap visualisation
colormap jet;
scale = 7;
joints = [1 3 5 8];
figure('Visible','off')
for j=joints
    map = unaries{scale}{j};
    big = imresize(map, 4);
    % This is the square root of the joint presence probabilities
    imagesc(sqrt(exp(big)));
    axis equal;
    axis off;
    filename = sprintf('heatmap-s%i-j%i', scale, j);
    print(filename, '-dpng', '-r300');
end
end

