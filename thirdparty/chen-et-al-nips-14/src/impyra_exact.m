function pyra = impyra_exact(im, model, upS)
% Compute feature pyramid.
%
% pyra.feat{i} is the i-th level of the feature pyramid.
% pyra.scales{i} is the scaling factor used for the i-th level.
% pyra.feat{i+interval} is computed at exactly half the resolution of feat{i}.
% first octave halucinates higher resolution data.
cnnpar = model.cnn;
psize = cnnpar.psize;
mean_pixel = single(cnnpar.mean_pixel);
mean_pixel = permute(mean_pixel(:), [3,2,1]);
cnn_output_dim = cnnpar.cnn_output_dim; %260;
batch_size = cnnpar.batch_size;%1000;

step = cnnpar.step;

interval = model.interval;

padx      = max(ceil((double(psize(1)-1)/2)),0); % more than half is visible
pady      = max(ceil((double(psize(2)-1)/2)),0); % more than half is visible
sc = 2 ^(1/interval);
imsize = [size(im, 1), size(im, 2)];
max_scale = 1 + floor(log(min(imsize)/max(psize))/log(sc));
impatchs = cell(max_scale,1);

scale = zeros(max_scale,1);
sizs = zeros(max_scale,2);

im = single(imresize(im,upS));  % may upscale image to better handle small objects.
% pyra is structure
pyra = struct('feat', cell(max_scale,1), 'sizs', cell(max_scale,1), 'scale', cell(max_scale, 1), ...
  'padx', cell(max_scale,1), 'pady', cell(max_scale,1));
% parfor i = 1:max_scale
for i = 1:max_scale
  scaled = imresize(im, 1/sc^(i-1));
  % PADDing the image
  scaled = padarray(scaled, [padx, pady, 0], 'replicate');
  [d1, d2, ~] = size(scaled);
  d1 = d1 - psize(2) + 1;
  d2 = d2 - psize(1) + 1;
  
  d1idx = 1:step:d1;
  d2idx = 1:step:d2;
  impatchs{i} = zeros(psize(2), psize(1), 3, numel(d1idx)*numel(d2idx), 'single');
  cnt = 0;
  for i2 = 1:numel(d2idx)
    for i1 = 1:numel(d1idx)
      cnt = cnt + 1;
      impatchs{i}(:,:,:,cnt) = ...
        bsxfun(@minus, scaled(d1idx(i1):d1idx(i1)+psize(2)-1,d2idx(i2):d2idx(i2)+psize(1)-1,:), mean_pixel);
    end
  end
  scale(i) = 1/sc^(i-1);
  sizs(i,:) = [numel(d1idx),numel(d2idx)];
end
impatchs = cat(4, impatchs{:});

bats = 1:cnnpar.batch_size:size(impatchs,4);
resp = cell(numel(bats),1);
input_data = zeros(cnnpar.psize(2), cnnpar.psize(1), 3, cnnpar.batch_size, 'single');
for bb = 1:numel(bats)
  i =  bats(bb);
  left = i;
  right = min(i+cnnpar.batch_size-1, size(impatchs,4));
  input_data(:,:,:,(left:right)-left+1) = impatchs(:,:,:,left:right);
  scores = caffe('conv_forward', {input_data});
  scores = double(reshape(scores{1}, [cnn_output_dim, batch_size]));
  resp{bb} = scores';
end
resp = cat(1, resp{:});

cnt = 0;
for i = 1:max_scale
  map_size = sizs(i,:);
  
  mapS = resp(cnt+1:cnt+map_size(1)*map_size(2),:);
  cnt = cnt + map_size(1)*map_size(2);
  
  mapS = reshape(mapS, [map_size(1),map_size(2), cnn_output_dim]);
  pyra(i).feat = mapS;
  pyra(i).sizs = sizs(i,:);
  pyra(i).scale = step ./ (upS * 1/sc^(i-1));
  pyra(i).pady = pady / step;
  pyra(i).padx = padx / step;
end


