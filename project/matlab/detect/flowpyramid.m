function flowpyra = flowpyramid(img1, img2, fim, model)
% Compute flow pyramid.
%
% pyra.feat{i} is the i-th level of the feature pyramid.
% pyra.scales{i} is the scaling factor used for the i-th level.
% pyra.feat{i+interval} is computed at exactly half the resolution of feat{i}.
% first octave halucinates higher resolution data.

sbin      = model.sbin;
interval  = model.interval;
padx      = max(model.maxsize(2)-1-1,0);
pady      = max(model.maxsize(1)-1-1,0);
sc = 2 ^(1/interval);
fimsize = [size(fim, 1) size(fim, 2)];
max_scale = 1 + floor(log(min(fimsize)/(5*sbin))/log(sc));
%pyra.feat = cell(max_scale,1);
flowpyra.scale = zeros(max_scale,1);

% if size(fim, 3) == 1
%   fim = repmat(fim,[1 1 3]);
% end
fim(:,:,3)=fim(:,:,1);
fim = double(fim); % our resize function wants floating point values
img1=double(img1);
img2=double(img2);

for i = 1:interval
  scaled = resize(fim, 1/sc^(i-1)); scaled = scaled/(sc^(i-1)); 
  
  scaled_im1 = resize(img1, 1/sc^(i-1)); 
  flowpyra.img1{i} = scaled_im1;
  scaled_im2 = resize(img2, 1/sc^(i-1));
  flowpyra.img2{i} =scaled_im2;
  
  flowpyra.flow{i} =  compute_flow_in_cell(scaled, sbin); %scaled; %features(scaled,sbin);
  flowpyra.scale(i) = 1/sc^(i-1);
  % remaining interals
  for j = i+interval:interval:max_scale
    scaled = reduce(scaled); scaled = scaled/2;
    
    scaled_im1 = reduce(scaled_im1); 
    flowpyra.img1{j} = scaled_im1;
    scaled_im2 = reduce(scaled_im2);
    flowpyra.img2{j} = scaled_im2;    
    
    flowpyra.flow{j} = compute_flow_in_cell(scaled, sbin); %scaled; %features(scaled,sbin);
    flowpyra.scale(j) = 0.5 * flowpyra.scale(j-interval);
  end
end

for i = 1:length(flowpyra.flow)
  % add 1 to padding because feature generation deletes a 1-cell
  % wide border around the feature map
  flowpyra.flow{i} = padarray(flowpyra.flow{i}, [pady+1 padx+1 0], 0);
  flowpyra.img1{i} = padarray(flowpyra.img1{i}, [pady+1 padx+1 0], 0);
  flowpyra.img2{i} = padarray(flowpyra.img2{i}, [pady+1 padx+1 0], 0);
  % write boundary occlusion feature
%   pyra.feat{i}(1:pady+1, :, end) = 1;
%   pyra.feat{i}(end-pady:end, :, end) = 1;
%   pyra.feat{i}(:, 1:padx+1, end) = 1;
%   pyra.feat{i}(:, end-padx:end, end) = 1;
end

flowpyra.scale    = model.sbin./flowpyra.scale;
flowpyra.interval = interval;
flowpyra.imy = fimsize(1);
flowpyra.imx = fimsize(2);
flowpyra.pady = pady;
flowpyra.padx = padx;
end

function F = compute_flow_in_cell(fim, bin)
    F1 = compute_flow_in_cell_internal(fim(:,:,1), bin);
    F2 = compute_flow_in_cell_internal(fim(:,:,2), bin);
    F = cat(3, F1,F2);
end


function F = compute_flow_in_cell_internal(fim, bin)
    [nr,nc]=size(fim);
    blocksy=round(nr/bin); blocksx=round(nc/bin);
    maxy=(blocksy-2)*bin; maxx=(blocksx-2)*bin; % this is how things are done for hog
    
    cr=1; cc=0; F = zeros(maxy/bin,maxx/bin);
    for r=bin:bin:maxy
        for c=bin:bin:maxx
            pat = fim(r:r+bin, c:c+bin);
            %[~,idx] = max(abs(pat(:)));
            cc=cc+1;
            F(cr,cc) = mean(pat(:)); %pat(idx);
        end
        cr = cr+1;
        cc = 0;
    end
    F = F/bin; % note that the entire image size has been shrunk by the cell size.
end

