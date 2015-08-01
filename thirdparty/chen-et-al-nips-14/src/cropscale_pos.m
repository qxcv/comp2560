function [im, box] = cropscale_pos(im, box, psize)
% Crop positive example to speed up latent search.

x1 = box.xy(:,1);
y1 = box.xy(:,2);
x2 = box.xy(:,3);
y2 = box.xy(:,4);
siz = x2(1)-x1(1)+1;
psize = psize(1);

x1 = min(x1); y1 = min(y1); x2 = max(x2); y2 = max(y2);

% crop image around bounding box
pad = siz * 0.5;
x1 = max(1, round(x1-pad));
y1 = max(1, round(y1-pad));
x2 = min(size(im,2), round(x2+pad));
y2 = min(size(im,1), round(y2+pad));

im = im(y1:y2, x1:x2, :);
box.xy(:,[1 3]) = box.xy(:,[1 3]) - x1 + 1;
box.xy(:,[2 4]) = box.xy(:,[2 4]) - y1 + 1;

% further scale it
sc = min(1, psize * 1.2 / siz); % keep it a little bit larger to make it more stable
im = imresize(im, sc);

box.xy = (box.xy - 1)*sc + 1;
