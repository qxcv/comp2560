function compile()
conf = global_conf();
caffe_root = conf.caffe_root;
if ~exist(caffe_root, 'dir')
  error('Please install Caffe in %s', caffe_root);
end
mexcmd = 'mex -outdir bin';
mexcmd = [mexcmd ' -O'];
mexcmd = [mexcmd ' -L/usr/lib -L/usr/local/lib'];

eval([mexcmd ' src/mex/distance_transform.cpp']);
eval([mexcmd ' external/qpsolver/qp_one_sparse.cc']);
eval([mexcmd ' external/qpsolver/score.cc']);
eval([mexcmd ' external/qpsolver/lincomb.cc']);

eval([mexcmd [' -I', fullfile(caffe_root, 'build/src/'), ' -lprotobuf -llmdb ', ...
  fullfile(caffe_root, '/build/lib/libcaffe.a'), ' src/mex/store_patch.cpp']]);
