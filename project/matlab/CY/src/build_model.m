function model = build_model(pa, clusters, tsize)
% jointmodel = buildmodel(name,model,pa,def,idx,K)
% This function merges together separate part models into a tree structure

conf = global_conf();
K = conf.K;

[nbh_IDs, global_IDs, target_IDs] = get_IDs(pa, K);

% parameters need for cnn
model.cnn = conf.cnn;
model.cnn.mean_pixel = compute_mean_pixel();
model.cnn.cnn_output_dim = global_IDs{end}(end)+1;   % +1 for background
model.cnn.step = conf.step;
model.cnn.psize = tsize * conf.step;
model.cnn.batch_size = conf.batch_size;              

model.tsize = tsize;
model.global_IDs = global_IDs;
model.nbh_IDs = nbh_IDs;
model.target_IDs = target_IDs;
model.K = K;

model.bias    = struct('w',{},'i',{});             % bias
model.apps = struct('w',{},'i',{});                % appearance of each part
model.pdefs = struct('w',{},'i',{});               % prior of deformation (regressed)
model.gaus    = struct('w',{},'i',{},'mean',{}, 'var', {});   % deformation gaussian

model.components{1} = struct('parent',{}, 'pid', {}, 'nbh_IDs', {}, ...
  'biasid',{},'appid',{},'app_global_ids',{},'pdefid',{},'gauid',{},'idpr_global_ids',{});

model.pa = pa;
model.tsize  = model.tsize;
model.interval = conf.interval;
model.sbin = conf.step;
model.len = 0;

% cnn parameters
model.cnn = model.cnn;

% add children
for i = 1:length(pa)
  child = i;
  parent = pa(child);
  assert(parent < child);
  p.parent = parent;
  p.pid    = child;
  p.nbh_IDs = nbh_IDs{p.pid};
  
  % add bias
  p.biasid = [];
  if parent == 0
    nb  = length(model.bias);
    b.w = 0;
    b.i = model.len + 1;
    model.bias(nb+1) = b;
    model.len = model.len + numel(b.w);
    p.biasid = nb+1;
  end
  
  % add appearance parameters
  p.appid = [];
  
  nf  = length(model.apps);
  f.w = 0.01;                     % encourage larger appearance score
  f.i = model.len + 1;
  
  model.apps(nf+1) = f;
  model.len = model.len + numel(f.w);
  
  p.appid = [p.appid, nf+1];
  p.app_global_ids = global_IDs{i};
  % add prior of deformation parameters
  p.pdefid = [];
  for nn = 1:numel(p.nbh_IDs)
    np = length(model.pdefs);
    pd.w = 0.01;                % encourage larger prior score
    pd.i = model.len + 1;
    
    model.pdefs(np+1) = pd;
    model.len = model.len + numel(pd.w);
    p.pdefid(nn) = np+1;
  end
  % add gaussian parameters (for spring deformation)
  p.gauid = [];
  p.idpr_global_ids = cell(numel(p.nbh_IDs), 1);
  for nn = 1:numel(p.nbh_IDs)
    idpr_idx = global_IDs{i};
    idpr_idx = permute(idpr_idx, [nn, 1:nn-1, nn+1:ndims(idpr_idx)]);
    p.idpr_global_ids{nn} = cell(numel(clusters{p.pid}{nn}), 1);
    for k = 1:numel(clusters{p.pid}{nn})    
      p.idpr_global_ids{nn}{k} = idpr_idx(k,:);
      center = clusters{p.pid}{nn}(k).center;
      %             sigma = clusters{p.pid}{nn}(k).sigma;
      sigma = [1,1];      % it seems 'sigma' does not improve performance
      ng  = length(model.gaus);
      g.w = [0.01, 0, 0.01, 0]; % [dx^2, dx, dy^2, dy]the normalization factor + variance for (x,y)
      % res = [-dx^2, -dx, -dy^2, -dy]';
      g.i = model.len + 1;
      g.mean = center;
      g.var = sigma;
      
      model.gaus(ng+1) = g;
      model.len = model.len + numel(g.w);
      p.gauid{nn}(k) = ng+1; % the kth mixture of deformation w.r.t nth neighbor
    end
  end
  
  np = length(model.components{1});
  model.components{1}(np+1) = p;
end


