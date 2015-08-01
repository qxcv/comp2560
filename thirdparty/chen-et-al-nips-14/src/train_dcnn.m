function train_dcnn(pos_train, pos_val, neg_train, tsize, caffe_solver_file)
conf = global_conf();
cachedir = conf.cachedir;
cnn = conf.cnn;
% -------------------------------------------------------------------------
% prepare training patches 
% -------------------------------------------------------------------------
prepare_patches(pos_train, pos_val, neg_train, tsize);
clear mex;
% -------------------------------------------------------------------------
% compute mean pixel
% -------------------------------------------------------------------------
if 1
  mean_pixel = [128, 128, 128];
  mean_pixel_file = [cachedir, 'mean_pixel.mat'];
  parsave(mean_pixel_file, mean_pixel);
else
  warning('Please update the mean pixel values in caffe .prototxt');
  mean_pixel = compute_mean_pixel();
  for ii = 1:3
    fprintf('Mean Pixel Chanel #%i: %.2f\n', ii, mean_pixel(ii));
  end
end

% -------------------------------------------------------------------------
% train dcnn
% -------------------------------------------------------------------------
caffe_root = conf.caffe_root;
model_file = cnn.cnn_model_file;
if ~exist(model_file, 'file')
  fprintf('Training model using gpu id: %d\n', conf.device_id);
  system([caffe_root, '/build/tools/caffe train ', sprintf('-gpu %d -solver %s', ...
    conf.device_id, caffe_solver_file)]);
end

% -------------------------------------------------------------------------
% dcnn net surgery
% -------------------------------------------------------------------------
% get fully-convolutional net
deploy_file = cnn.cnn_deploy_file;
fully_conv_model_file = cnn.cnn_conv_model_file;
deploy_conv_file = cnn.cnn_deploy_conv_file;
if ~exist(fully_conv_model_file, 'file')
  net_surgery(model_file, deploy_file, fully_conv_model_file, deploy_conv_file);
  fprintf('fully convolutional model saved as %s\n', fully_conv_model_file);
end
