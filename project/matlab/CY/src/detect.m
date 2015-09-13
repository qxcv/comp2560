function [boxes,model,ex] = detect(iminfo, model, thresh, bbox, overlap, id, label)
% Detect objects in image using a model and a score threshold.
% Higher threshold leads to fewer detections.
%
% The function returns a matrix with one row per detected object.  The
% last column of each row gives the score of the detection.  The
% column before last specifies the component used for the detection.
% Each set of the first 4 columns specify the bounding box for a part
%
% If bbox is not empty, we pick best detection with significant overlap.
% If label is included, we write feature vectors to a global QP structure
%
% This function updates the model (by running the QP solver) if upper and lower bound differs

INF = 1e10;
conf = global_conf();
useGpu = conf.useGpu;

if nargin > 3 && ~isempty(bbox)
  latent = true;
  if label > 0
    thresh = -INF;
  end
else
  latent = false;
end

% Compute the feature pyramid and prepare filter
im = imreadx(iminfo);
% if has box information, crop it
if latent && label > 0
  % crop positive images to speed up latent search
  [im, bbox] = cropscale_pos(im, bbox, model.cnn.psize);
end

[pyra, unary_map, idpr_map] = imCNNdet(im,model,useGpu);

levels = 1:length(pyra);

% Define global QP if we are writing features
% Randomize order to increase effectiveness of model updating
write = false;
if nargin > 5
  global qp;
  write  = true;
  levels = levels(randperm(length(levels)));
end
if nargin < 6
  id = 0;
end
if nargin < 7
  label = 0;
end

% Cache various statistics derived from model
[components,apps] = modelcomponents(model);

boxes = zeros(100000,length(components{1})*4+2);
cnt = 0;

ex.blocks = [];
ex.id = [label id 0 0 0];

if latent && label > 0
  % record best when doing latent on positive example
  best_ex = ex;
  best_box = [];
end

% Iterate over random permutation of scales and components,
for level = levels
  % Iterate through mixture components
  sizs = pyra(level).sizs;
  for c  = randperm(length(model.components))
    parts = components{c};
    p_no = length(parts);
    
    % Skip if there is no overlap of root filter with bbox
    if latent
      skipflag = 0;
      for p = 1:p_no
        % because all mixtures for one part is the same size, we only need to do this once
        ovmask = testoverlap(parts(p).sizx(1),parts(p).sizy(1),sizs(1),sizs(2),pyra(level),bbox.xy(p,:),overlap);
        if ~any(ovmask)
          skipflag = 1;
          break;
        end
      end
      if skipflag == 1
        continue;
      end
    end
    % Local scores
    
    for p = 1:p_no
      % assign each deformation scores
      parts(p).defMap = idpr_map{level}{p};
      % --------------
      parts(p).appMap = unary_map{level}{p};
      
      f = parts(p).appid;
      parts(p).score = parts(p).appMap * apps{f};
      
      parts(p).level = level;
      
      if latent
        ovmask = testoverlap(parts(p).sizx,parts(p).sizy,sizs(1),sizs(2),pyra(level),bbox.xy(p,:),overlap);
        tmpscore = parts(p).score;
        % label supervision
        if label > 0
          tmpscore(~ovmask) = -INF;
          % mixture part supervision
          if isfield(iminfo,'near')
            for n = 1:numel(parts(p).nbh_IDs)
              parts(p).defMap{n}(:,:,~iminfo.near{p}{n}) = -INF;
            end
          end
        elseif label < 0
          tmpscore(ovmask) = -INF;
        end
        parts(p).score = tmpscore;
      end
    end
    
    % Walk from leaves to root of tree, passing message to parent
    for p = p_no:-1:2
      child = parts(p);
      par = parts(p).parent;
      parent = parts(par);
      cbid = find(child.nbh_IDs == parent.pid);
      pbid = find(parent.nbh_IDs == child.pid);
      
      [msg,parts(p).Ix,parts(p).Iy,parts(p).Im{cbid},parts(par).Im{pbid}] ...
        = passmsg(child, parent, cbid, pbid);
      parts(par).score = parts(par).score + msg;
    end
    
    % Add bias to root score
    parts(1).score = parts(1).score + parts(1).b;
    rscore = parts(1).score;
    
    % keep the positive example with the highest score in latent mode
    if latent && label > 0
      thresh = max(thresh,max(rscore(:)));
    end
    
    [Y,X] = find(rscore >= thresh);
    % Walk back down tree following pointers
    % (DEBUG) Assert extracted feature re-produces score
    for i = 1:length(X)
      cnt = cnt + 1;
      x = X(i);
      y = Y(i);
      
      [box,ex] = backtrack(x,y,parts,pyra(level),ex,write);
      
      boxes(cnt,:) = [box c rscore(y,x)];
      if write && (~latent || label < 0)
        qp_write(ex);
        qp.ub = qp.ub + qp.Cneg*max(1+rscore(y,x),0);
      elseif latent && label > 0
        if isempty(best_box)
          best_box = boxes(cnt,:);
          best_ex = ex;
        elseif best_box(end) < rscore(y,x)
          % update best
          best_box = boxes(cnt,:);
          best_ex = ex;
        end
      end
    end
    
    % Crucial DEBUG assertion:
    % If we're computing features, assert extracted feature re-produces score
    % (see qp_writ.m for computing original score)
    if write && (~latent || label < 0) && ~isempty(X) && qp.n < length(qp.a)
      w = -(qp.w + qp.w0.*qp.wreg) / qp.Cneg;
      assert(abs(score(w,qp.x,qp.n) - rscore(y,x)) < 1e-5);
    end
    
    % Optimize qp with coordinate descent, and update model
    if write && (~latent || label < 0) && ...
        (qp.lb < 0 || 1 - qp.lb/qp.ub > .05 || qp.n == length(qp.sv))
      model = optimize(model);
      [components,apps] = modelcomponents(model);
    end
  end
end

boxes = boxes(1:cnt,:);

if latent && ~isempty(boxes) && label > 0
  boxes = best_box;
  if write
    qp_write(best_ex);
  end
end

% Backtrack through dynamic programming messages to estimate part locations
% and the associated feature vector
function [box,ex] = backtrack(x,y,parts,pyra,ex,write)
numparts = length(parts);
ptr = zeros(numparts,2);
box = zeros(numparts,4);
k   = 1;
p   = parts(k);
ptr(k,:) = [x,y];
scale = pyra.scale;
x1  = (x - 1 - pyra.padx)*scale+1;
y1  = (y - 1 - pyra.pady)*scale+1;
x2  = x1 + p.sizx*scale - 1;
y2  = y1 + p.sizy*scale - 1;

box(k,:) = [x1 y1 x2 y2];

if write
  ex.id(3:5) = [p.level round(x+p.sizx/2) round(y+p.sizy/2)];
  ex.blocks = [];
  ex.blocks(end+1).i = p.biasI;
  ex.blocks(end).x   = 1;
  f = parts(k).appMap(y, x);
  ex.blocks(end+1).i = p.appI;
  ex.blocks(end).x   = f;
end
for k = 2:numparts
  p   = parts(k);
  par = p.parent;
  
  x   = ptr(par,1);
  y   = ptr(par,2);
  
  ptr(k,1) = p.Ix(y,x);
  ptr(k,2) = p.Iy(y,x);
  
  x1  = (ptr(k,1) - 1 - pyra.padx)*scale+1;
  y1  = (ptr(k,2) - 1 - pyra.pady)*scale+1;
  x2  = x1 + p.sizx*scale - 1;
  y2  = y1 + p.sizy*scale - 1;
  box(k,:) = [x1 y1 x2 y2];
  
  if write
    cbid = find(p.nbh_IDs == parts(par).pid);
    pbid = find(parts(par).nbh_IDs == p.pid);
    
    cm = p.Im{cbid}(y,x);
    pm = parts(par).Im{pbid}(y,x);
    
    % two prior of deformation
    ex.blocks(end+1).i = p.pdefI(cbid);
    ex.blocks(end).x = p.defMap{cbid}(ptr(k,2),ptr(k,1),cm);
    
    ex.blocks(end+1).i = parts(par).pdefI(pbid);
    ex.blocks(end).x = parts(par).defMap{pbid}(y,x,pm);
    % two deformations
    ex.blocks(end+1).i = p.gauI{cbid}(cm);
    ex.blocks(end).x   = defvector(p, ptr(k,1),ptr(k,2),x,y,cm,cbid);
    
    ex.blocks(end+1).i = parts(par).gauI{pbid}(pm);
    ex.blocks(end).x   = defvector(parts(par),x,y,ptr(k,1),ptr(k,2),pm,pbid);
    
    x   = ptr(k,1);
    y   = ptr(k,2);
    
    % unary
    f   = parts(k).appMap(y,x);
    ex.blocks(end+1).i = p.appI;
    ex.blocks(end).x = f;
  end
end
box = reshape(box',1,4*numparts);

% Update QP with coordinate descent
% and return the asociated model
function model = optimize(model)

global qp;
fprintf('.');
if qp.lb < 0 || qp.n == length(qp.a),
  qp_opt();
  qp_prune();
else
  qp_one();
end
model = vec2model(qp_w(),model);

