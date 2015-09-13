
% =============
% Detection code
if isunix()
  cd ./YR/mex_unix
  % use one of the following depending on your setup
  % 1 is fastest, 3 is slowest 
  % 1) multithreaded convolution using blas
 % mex -O fconvblas.cc -lmwblas fconv
  % 2) mulththreaded convolution without blas
  % mex -O fconvMT.cc -o fconv 
  % 3) basic convolution, very compatible
  % mex -O fconv.cc -o fconv
elseif ispc()
  cd ./YR/mex_pc;
%  mex -O fconv.cc
end

% for pc,  you need to specify the paths of the include files!
mex -O fconv.cc
mex -O resize.cc
mex -O reduce.cc
mex -O dt.cc
mex -O shiftdt.cc
mex -O features.cc

cd ../../;
