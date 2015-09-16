function CY_startup
if ~exist('./CY/bin', 'dir')
  mkdir('./CY/bin');
end

if ~isdeployed
  addpath('./CY/dataio');
  addpath('./CY/bin');
  addpath('./CY/evaluation');
  addpath('./CY/visualization');
  addpath('./CY/src');
  addpath('./CY/tools');
  addpath('./CY/external');
  % path to DCNN library, e.g., caffe
  conf = global_conf();
  caffe_root = conf.caffe_root;
  if exist(fullfile(caffe_root, '/matlab/caffe'), 'dir')
    addpath(fullfile(caffe_root, '/matlab/caffe'));
  else
    warning('Please install Caffe in %s', caffe_root);
  end
end
