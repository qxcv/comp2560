function model = train(name, model, pos, neg, iter, C, wpos, maxsize, overlap)
% Train a structured SVM with latent assignement of positive variables
% pos  = list of positive images with part annotations
% neg  = list of negative images
% iter is the number of training iterations
%   C  = scale factor for slack loss
% wpos =  amount to weight errors on positives
% maxsize = maximum size of the training data cache (in GB)
% overlap =  minimum overlap in latent positive search

model.name = name;
conf = global_conf();

if ~exist('iter', 'var')
  iter = 1;
end

if ~exist('C', 'var')
  C = 0.002;
end

if ~exist('wpos', 'var')
  wpos = 2;
end

if ~exist('maxsize', 'var')
  % Estimated #sv = (wpos + 1) * # of positive examples
  % maxsize*1e9/(4*model.len)  = # of examples we can store, encoded as 4-byte floats
  no_sv = (wpos+1) * length(pos);
  maxsize = 10 * no_sv * 4 * sparselen(model) / 1e9;
  maxsize = min(max(maxsize,conf.memsize),7.5);
end

fprintf('Using %.3f GB\n', maxsize);

if ~exist('overlap', 'var')
  overlap = 0.6; % at least 0.6 overlap to be positive
end

% Vectorize the model
len  = sparselen(model);
nmax = round(maxsize*.25e9/len);

% reset random seed to reproduce result
rng(0);

% Define global QP problem
clear global qp;
global qp;
% qp.x(:,i) = examples
% qp.i(:,i) = id
% qp.b(:,i) = bias of linear constraint
% qp.d(i)   = ||qp.x(:,i)||^2
% qp.a(i)   = ith dual variable
qp.x   = zeros(len,nmax,'single');
qp.i   = zeros(5,nmax,'int32');
qp.b   = ones(nmax,1,'single');
qp.d   = zeros(nmax,1,'double');
qp.a   = zeros(nmax,1,'double');
qp.sv  = false(1,nmax);
qp.n   = 0;
qp.lb = [];

[qp.w,qp.wreg,qp.w0,qp.noneg] = model2vec(model);
qp.Cpos = C*wpos;
qp.Cneg = C;
qp.w    = (qp.w - qp.w0).*qp.wreg;

for t = 1:iter,
  fprintf('\niter: %d/%d\n', t, iter);
  
  qp.n = 0;
  numpositives = poslatent(name, t, model, pos, overlap);
  
  for i = 1:length(numpositives),
    fprintf('component %d got %d positives\n', i, numpositives(i));
  end
  assert(qp.n <= nmax);
  
  % Fix positive examples as permanent support vectors
  % Initialize QP soln to a valid weight vector
  % Update QP with coordinate descent
  qp.svfix = 1:qp.n;
  qp.sv(qp.svfix) = 1;
  
  qp_prune();
  qp_opt();
  model = vec2model(qp_w(),model);
  
  % grab negative examples from negative images
  mining_onneg(model, neg, nmax)
  
  % One final pass of optimization
  qp_opt();
  model = vec2model(qp_w(),model);
  
  fprintf('\nDONE iter: %d/%d #sv=%d/%d, LB=%.4f\n',t,iter,sum(qp.sv),nmax,qp.lb);
  
  % Compute minimum score on positive example (with raw, unscaled features)
  r = sort(qp_scorepos());
  model.thresh   = r(ceil(length(r)*.05));
  model.lb = qp.lb;
  model.ub = qp.ub;
end
fprintf('qp.x size = [%d %d]\n',size(qp.x));
clear global qp;

% negative mining on negative images
function numnegatives = mining_onneg(model, neg, nmax)
model.interval = 3;
numnegatives = 0;
global qp;
% grab negative examples from negative images
for i = 1:length(neg)
  
  fprintf('\n Image(%d/%d)',i,length(neg));
  [box,model] = detect(neg(i), model, -1, [], 0, i, -1);
  numnegatives = numnegatives + size(box,1);
  fprintf(' #cache+%d=%d/%d, #sv=%d, #sv>0=%d, (est)UB=%.4f, LB=%.4f',size(box,1),qp.n,nmax,sum(qp.sv),sum(qp.a>0),qp.ub,qp.lb);
  
  % Stop if cache is full
  if sum(qp.sv) == nmax,
    break;
  end
end

% get positive examples using latent detections
% we create virtual examples by flipping each image left to right
function numpositives = poslatent(name, t, model, pos, overlap)
numpos = length(pos);
numpositives = zeros(length(model.components), 1);
minsize = prod(double(model.tsize*model.sbin));

for ii = 1:numpos
  fprintf('%s: iter %d: latent positive: %d/%d\n', name, t, ii, numpos);
  % skip small examples
  scale_x = pos(ii).scale_x; scale_y = pos(ii).scale_y;
  bbox.xy = [pos(ii).joints(:,1)-scale_x, pos(ii).joints(:,2)-scale_y, ...
    pos(ii).joints(:,1)+scale_x, pos(ii).joints(:,2)+scale_y];
  area = (bbox.xy(:,3)-bbox.xy(:,1)+1).*(bbox.xy(:,4)-bbox.xy(:,2)+1);
  if any(area < minsize/1.5)
    % skip only when exmaple are too small
    continue;
  end
  
  % get example
  box = detect(pos(ii), model, 0, bbox, overlap, ii, 1);
  if ~isempty(box),
    fprintf(' (comp=%d,sc=%.3f)\n',box(1,end-1),box(1,end));
    c = box(1,end-1);
    numpositives(c) = numpositives(c)+1;
  end
end

% Compute score (weights*x) on positives examples (see qp_write.m)
% Standardized QP stores w*x' where w = (weights-w0)*r, x' = c_i*(x/r)
% (w/r + w0)*(x'*r/c_i) = (v + w0*r)*x'/ C
function scores = qp_scorepos

global qp;
y = qp.i(1,1:qp.n);
I = find(y == 1);
w = qp.w + qp.w0.*qp.wreg;
scores = score(w,qp.x,I) / qp.Cpos;

% Computes expected number of nonzeros in sparse feature vector
function len = sparselen(model)

numblocks = 0;
for c = 1:length(model.components)
  feat = zeros(model.len,1);
  for p = model.components{c},
    if ~isempty(p.biasid)
      x = model.bias(p.biasid(1));    % use only one biase
      i1 = x.i;
      i2 = i1 + numel(x.w) - 1;
      feat(i1:i2) = 1;
      numblocks = numblocks + 1;
    end
    if ~isempty(p.appid)
      x  = model.apps(p.appid(1));  % use only one appearance filter
      i1 = x.i;
      i2 = i1 + numel(x.w) - 1;
      feat(i1:i2) = 1;
      numblocks = numblocks + 1;
    end
    nbh_N = numel(p.nbh_IDs);
    if ~isempty(p.pdefid)
      for i = 1:nbh_N
        x = model.pdefs(p.pdefid(i));       % use one deformation prior for each neighbor
        i1 = x.i;
        i2 = i1 + numel(x.w) - 1;
        feat(i1:i2) = 1;
        numblocks = numblocks + 1;
      end
    end
    if ~isempty(p.gauid)
      for i = 1:nbh_N
        x  = model.gaus(p.gauid{i}(1));    % use only one kind of deformation in each mixture
        i1 = x.i;
        i2 = i1 + numel(x.w) - 1;
        feat(i1:i2) = 1;
        numblocks = numblocks + 1;
      end
    end
  end
  
  % Number of entries needed to encode a block-sparse representation
  %   1 + numberofblocks*2 + #nonzeronumbers
  len = 1 + numblocks*2 + sum(feat);
end


