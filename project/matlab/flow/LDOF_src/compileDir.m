root_dir = pwd;
flow_dir=[root_dir '/flow/LDOF_src/'];

cd([flow_dir '/src/']);
mex sor_warping_flow_multichannel_LDOF.cpp;

cd([flow_dir '/third_party/ann_mwrapper/']);
ann_compile_mex;

cd(root_dir);
