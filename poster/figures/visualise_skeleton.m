% Displays skeletons from boxes (spooky)
function visualise_skeleton(img, boxes, max_to_save)
parent = [0 1 2 3 4 5 6 3 8 9 2 11 12 13 14 11 16 17];

if nargin < 3
    max_to_save = 1;
end

for i = 1:length(parent)
    x1(:,i) = boxes(:,1+(i-1)*4);
    y1(:,i) = boxes(:,2+(i-1)*4);
    x2(:,i) = boxes(:,3+(i-1)*4);
    y2(:,i) = boxes(:,4+(i-1)*4);
end
x = (x1 + x2)/2;
y = (y1 + y2)/2;

figure('Visible','off');
for b=1:min(max_to_save, length(boxes))
    imshow(img); hold on;
    line([x(b,3), x(b,5)], [y(b,3), y(b,5)],'color', 'r','linewidth', 3);
    line([x(b,5), x(b,7)], [y(b,5), y(b,7)],'color', 'b','linewidth', 3);
    line([x(b,11), x(b,13)], [y(1,11), y(b,13)],'color', 'g','linewidth', 3);
    line([x(b,13), x(b,15)], [y(b,13), y(b,15)],'color', 'y','linewidth', 3);
    % Neck
    line([x(b,1), x(b,2)], [y(b,1), y(b,2)],'color', 'm','linewidth', 3);
    % Shoulders
    line([x(b,2), x(b,3)], [y(b,2), y(b,3)],'color', 'c','linewidth', 3);
    line([x(b,2), x(b,11)], [y(b,2), y(b,11)],'color', 'w','linewidth', 3);
    hold off;
    axis equal;
    axis off;
    name = sprintf('det-pose-%i', b);
    print(name, '-r300', '-dpng');
end
end

