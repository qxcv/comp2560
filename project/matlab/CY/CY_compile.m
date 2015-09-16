function CY_compile
conf = global_conf();
caffe_root = conf.caffe_root;
if ~exist(caffe_root, 'dir')
  error('Please install Caffe in %s', caffe_root);
end
mexcmd = 'mex -outdir CY/bin';
mexcmd = [mexcmd ' -O'];
mexcmd = [mexcmd ' -L/usr/lib -L/usr/local/lib'];

eval([mexcmd ' CY/src/mex/distance_transform.cpp']);
eval([mexcmd [' -I', fullfile(caffe_root, 'build/src/'), ' -lprotobuf -llmdb ', ...
  fullfile(caffe_root, '/build/lib/libcaffe.a'), ' CY/src/mex/store_patch.cpp']]);
