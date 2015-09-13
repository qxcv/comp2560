function pyra = impyra(im, model, upS)
% Compute feature pyramid.
%
% pyra.feat{i} is the i-th level of the feature pyramid.
% pyra.scales{i} is the scaling factor used for the i-th level.
% pyra.feat{i+interval} is computed at exactly half the resolution of feat{i}.
% first octave halucinates higher resolution data.
cnnpar = model.cnn;
psize = cnnpar.psize;
if isfield(cnnpar, 'mean_pixel')                    % for compatible
  mean_pixel = single(cnnpar.mean_pixel);
  mean_pixel = permute(mean_pixel(:), [3,2,1]);
else
  mean_pixel(1,1,:) = single([128,128,128]);
end

im = single(imresize(im,upS));  % may upscale image to better handle small objects.

step = cnnpar.step;

interval = model.interval;

padx      = max(ceil((double(psize(1)-1)/2)),0); % more than half is visible
pady      = max(ceil((double(psize(2)-1)/2)),0); % more than half is visible
sc = 2 ^(1/interval);
imsize = [size(im, 1), size(im, 2)];
max_scale = 1 + floor(log(min(imsize)/max(psize))/log(sc));

% pyra is structure
pyra = struct('feat', cell(max_scale,1), 'sizs', cell(max_scale,1), 'scale', cell(max_scale, 1), ...
  'padx', cell(max_scale,1), 'pady', cell(max_scale,1));

ibatch = interval;    % use smaller ibatch if out of memory
for i = 1:ibatch:max_scale
  scaled = imresize(im, 1/sc^(i-1));
  
  num = min(ibatch, max_scale-i+1);
  impyra = zeros(size(scaled,1)+2*padx, size(scaled,2)+2*pady, 3, num, 'single');
  for n = 0:num-1
    % the image
    scaled_pad = padarray(scaled, [padx, pady, 0], 'replicate');
    %             scaled_pad = padarray(scaled, [padx, pady, 0], 0);
    scaled_pad = bsxfun(@minus, scaled_pad, mean_pixel);
    impyra(1:size(scaled_pad,1), 1:size(scaled_pad,2),:,n+1) = scaled_pad;
    % output size
    pyra(i+n).sizs = floor([size(scaled_pad,1)-psize(1), size(scaled_pad,2)-psize(2)] / step) + 1;      % caffe -> ceil
    
    pyra(i+n).scale = step ./ (upS * 1/sc^(i-1+n));
    pyra(i+n).pady = pady / step;
    pyra(i+n).padx = padx / step;
    scaled = imresize(scaled, 1/sc);
  end
  resp = caffe('conv_forward', {impyra});      % softmax apply in caffe model.
  resp = resp{1};

  for n = 0:num-1
    pyra(i+n).feat = resp(1:pyra(i+n).sizs(1), 1:pyra(i+n).sizs(2), :, n+1);
  end
end

