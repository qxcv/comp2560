function [pyra, unary_map, idpr_map] = imCNNdet(im, model, useGpu, upS, impyra_fun)
if ~exist('useGpu', 'var')
  useGpu = 1;
end
if ~exist('upS', 'var')
  upS = 1;        % by default, we do not upscale the image
end
if ~exist('impyra_fun', 'var')
  impyra_fun = @impyra;
end
cnnpar = model.cnn;

% init caffe network (spews logging info)

if caffe('is_initialized') == 0
  if ~exist(cnnpar.cnn_deploy_conv_file, 'file') || ~exist(cnnpar.cnn_conv_model_file, 'file')
    error('model files not exist');
  end
  caffe('init', cnnpar.cnn_deploy_conv_file, cnnpar.cnn_conv_model_file);
  % set to use GPU or CPU
  if useGpu
    caffe('set_mode_gpu');
  else
    caffe('set_mode_cpu');
  end
  % put into test mode
  caffe('set_phase_test');
end

%
if upS > 1
  % ensure largest length < 1200
  [imx, imy, ~] = size(im);
  upS = min(upS, 600 / max(imx,imy));
end
%
pyra = impyra_fun(im, model, upS);
max_scale = numel(pyra);
FLT_MIN = realmin('single');
% 0.01;

nbh_IDs = model.nbh_IDs;
K = model.K;
unary_map = cell(max_scale, 1);
idpr_map = cell(max_scale, 1);
p_no = numel(nbh_IDs);
model_parts = model.components{1};

for i = 1:max_scale
  joint_prob = pyra(i).feat;
  % the first dimension is the reponse of background
  joint_prob = joint_prob(:,:,2:end);
  
  % marginalize
  unary_map{i} = cell(p_no, 1);
  idpr_map{i} = cell(p_no, 1);
  for p = 1:p_no
    app_global_ids = model_parts(p).app_global_ids;
    idpr_global_ids = model_parts(p).idpr_global_ids;
    nbh_N = numel(nbh_IDs{p});
    unary_map{i}{p} = sum(joint_prob(:,:,app_global_ids), 3);
    % convert to log space
    unary_map{i}{p} = log(max(unary_map{i}{p}, FLT_MIN));
    idpr_map{i}{p} = cell(nbh_N,1);
    
    for n = 1:nbh_N
      idpr_map{i}{p}{n} = zeros(size(joint_prob,1), size(joint_prob,2), K);               
      for m = 1:K
        idpr_map{i}{p}{n}(:,:,m) = sum(joint_prob(:,:,idpr_global_ids{n}{m}),3);
      end
      % normalize
      idpr_map{i}{p}{n} = idpr_map{i}{p}{n} ./ repmat(sum(idpr_map{i}{p}{n},3), [1,1,K]);     
      
      % convert to log space
      idpr_map{i}{p}{n} = log(max(idpr_map{i}{p}{n}, FLT_MIN));
    end
  end
end


