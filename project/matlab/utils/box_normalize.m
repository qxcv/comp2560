% normalize the boxes using the image size.
function box = box_normalize(box, imsize)
box(:,1:2:end) = box(:, 1:2:end)/imsize(2);
box(:,2:2:end) = box(:, 2:2:end)/imsize(1);
end