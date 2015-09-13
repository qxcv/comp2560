function startup
if ~exist('./bin', 'dir')
  mkdir('./bin');
end

if ~isdeployed
  addpath('./dataio');
  addpath('./bin');
  addpath('./evaluation');
  addpath('./visualization');
  addpath('./src');
  addpath('./tools');
  addpath('./external');
  addpath('./external/qpsolver');
  % path to DCNN library, e.g., caffe
  conf = global_conf();
  caffe_root = conf.caffe_root;
  if exist(fullfile(caffe_root, '/matlab/caffe'), 'dir')
    addpath(fullfile(caffe_root, '/matlab/caffe'));
  else
    warning('Please install Caffe in %s', caffe_root);
  end
end
