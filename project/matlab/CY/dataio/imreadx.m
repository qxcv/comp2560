function im = imreadx(ex)
im = make_color(imread(ex.im));
% !!! always flip first !!!!
if ex.isflip
  im = im(:,end:-1:1,:);
end
% rotate and flip
if ex.r_degree ~= 0
  im = imrotate(im, ex.r_degree);
end

function im = make_color(input)
% Convert input image to color.
%   im = color(input)

if size(input, 3) == 1
  im(:,:,1) = input;
  im(:,:,2) = input;
  im(:,:,3) = input;
else
  im = input;
end
