function boxes = detect_MM(im, model,thresh,partIDs)

% This algorithm  approximates nbest maximal decoders introduced in the paper,
% returning high-scoring and diverse poses.
% Greedy NMS is applied to max-marginal score map for each part.
% The poses are obtained by backtracking from the location returned by NMS.
% Authors : Dennis Park, Deva Ramanan {iypark,dramanan}@ics.uci.edu

% Compute the feature pyramid and prepare filter
pyra     = featpyramid(im,model);
interval = model.interval;
levels   = 1:length(pyra.feat);

[components,filters,resp] = modelcomponents(model,pyra);
boxes = [];

% Iterate over scales and components,
for rlevel = levels,
  for c = 1:length(model.components),
    cnt = 0;
    parts    = components{c};
    numparts = length(parts);
    boxes_t  = zeros(100000,numparts*4+2);
  
    % Local scores
    for k = 1:numparts,
      f     = parts(k).filterid;
      level = rlevel-parts(k).scale*interval;
      if isempty(resp{level}),
        resp{level} = fconv(pyra.feat{level},filters,1,length(filters));
      end
      for m = 1:length(f)
        parts(k).score(:,:,m) = resp{level}{f(m)};
      end
      parts(k).level = level;
    end
    
    % Walk from leaves to root of tree, passing message to parent
    for k = numparts:-1:2,
      par = parts(k).parent;
      [msg, Ix, Iy, Ik] = passmsg(parts(k),parts(par));
      parts(par).score = parts(par).score + msg;
      parts(k).msg = msg;
      parts(k).Ix = Ix;
      parts(k).Iy = Iy;
      parts(k).Ik = Ik;
    end
  
    % Add bias to root score
    parts(1).mm = parts(1).score + parts(1).b;
    
    % Walk down from root to leaves, computing max-marginals.
    for k = 2:numparts
      par = parts(k).parent;
      [msg_d, Ixd, Iyd, Ikd] = passmsg_d(parts(par),parts(k));
      parts(k).mm = parts(k).score + msg_d;
      parts(k).Ixd = Ixd;
      parts(k).Iyd = Iyd;
      parts(k).Ikd = Ikd;
    end

    % Backtrack from given partIDs
    for k = partIDs
      [score mix]= max(parts(k).mm,[],3);	
      %[Y,X] = find(score > thresh);
      [Y,X] = nms_loop(score,thresh);
      if length(X) > 0
        I = (X-1)*size(score,1) + Y;
        box = backtrack(X,Y,mix(I),k,parts,pyra);
        i   = cnt+1:cnt+length(I);
        boxes_t(i,:) = [box repmat(c,length(I),1) score(I)];
        cnt = i(end);
      end
    end
    boxes = cat(1,boxes,unique(boxes_t(1:cnt,:),'rows'));
  end
end

function [Y,X] = nms_loop(score,thresh,r)

% iterate over the following
% 1) find the location of highest score
% 2) suppress (r+1)x(r+1) block around the location

if nargin < 3
  r = 2;
end

Y = zeros(numel(score),1);
X = zeros(numel(score),1);
cnt = 0;
siz = size(score);

[ss ys] = max(score);
[s x]   = max(ss);
y       = ys(x);
while s >= thresh
  cnt = cnt + 1;
  Y(cnt) = y; 
  X(cnt) = x;
  x1 = max(x-r,1);
  y1 = max(y-r,1);
  x2 = min(x+r,size(score,2));
  y2 = min(y+r,size(score,1));
  score(y1:y2,x1:x2) = -inf;
  [ss ys] = max(score);
  [s x]   = max(ss);
  y       = ys(x);
end

Y = Y(1:cnt);
X = X(1:cnt);

function box = backtrack(x,y,mix,rt,parts,pyra)

  numx     = length(x);
  numparts = length(parts);
  
  xptr = zeros(numx,numparts);
  yptr = zeros(numx,numparts);
  mptr = zeros(numx,numparts);
  box  = zeros(numx,4,numparts);

  % reorder part indices so that "rt" is new root
  [list downptr] = reorder(rt,parts); 
  for i = 1:length(list)
    k = list(i);
    if k == rt
      xptr(:,k) = x;
      yptr(:,k) = y;
      mptr(:,k) = mix;
    else
      if downptr(i)
        chld = list(i-1);
        p    = parts(chld);
        assert(parts(chld).parent == k);
        [h,w,foo] = size(p.Ixd);
        I   = (mptr(:,chld)-1)*h*w + (xptr(:,chld)-1)*h + yptr(:,chld);
        xptr(:,k) = p.Ixd(I);
        yptr(:,k) = p.Iyd(I);
        mptr(:,k) = p.Ikd(I);
      else
        p   = parts(k);
        par = p.parent;
        [h,w,foo] = size(p.Ix);
        I   = (mptr(:,par)-1)*h*w + (xptr(:,par)-1)*h + yptr(:,par);
        xptr(:,k) = p.Ix(I);
        yptr(:,k) = p.Iy(I);
        mptr(:,k) = p.Ik(I);
      end
    end
    scale = pyra.scale(parts(k).level);
    x1 = (xptr(:,k) - 1 - pyra.padx)*scale+1;
    y1 = (yptr(:,k) - 1 - pyra.pady)*scale+1;
    x2 = x1 + parts(k).sizx(mptr(:,k))*scale - 1;
    y2 = y1 + parts(k).sizy(mptr(:,k))*scale - 1;
    box(:,:,k) = [x1 y1 x2 y2];
  end
  
  box = reshape(box,numx,4*numparts);

function [list down] = reorder(k,parts)
  
  numparts = length(parts);
  down     = false(1,numparts);
  rem      = true(1,numparts);
  rem(k)   = 0;
  
  list = k;
  par  = parts(k).parent;
  i = 1;
  while par ~= 0
    list = [list par];
    i = i + 1;
    down(i)  = 1;
    rem(par) = 0;
    k = par;
    par = parts(k).parent;
  end
  
  list = [list find(rem)];


% Cache various statistics from the model data structure for later use  
function [components,filters,resp] = modelcomponents(model,pyra)
  components = cell(length(model.components),1);
  for c = 1:length(model.components),
    for k = 1:length(model.components{c}),
      p = model.components{c}(k);
      [p.w,p.defI,p.starty,p.startx,p.step,p.level,p.Ix,p.Iy] = deal([]);
      [p.scale,p.level,p.Ix,p.Iy] = deal(0);
      
      % store the scale of each part relative to the component root
      par = p.parent;      
      assert(par < k);
      p.b = [model.bias(p.biasid).w];
      p.b = reshape(p.b,[1 size(p.biasid)]);
      p.biasI = [model.bias(p.biasid).i];
      p.biasI = reshape(p.biasI,size(p.biasid));
      p.sizx  = zeros(length(p.filterid),1);
      p.sizy  = zeros(length(p.filterid),1);
      
      for f = 1:length(p.filterid)
        x = model.filters(p.filterid(f));
        [p.sizy(f) p.sizx(f) foo] = size(x.w);
        p.filterI(f) = x.i;
      end
      for f = 1:length(p.defid)	  
        x = model.defs(p.defid(f));
        p.w(:,f)  = x.w';
        p.defI(f) = x.i;
        ax  = x.anchor(1);
        ay  = x.anchor(2);    
        ds  = x.anchor(3);
        p.scale = ds + components{c}(par).scale;
        % amount of (virtual) padding to hallucinate
        step     = 2^ds;
        virtpady = (step-1)*pyra.pady;
        virtpadx = (step-1)*pyra.padx;
        % starting points (simulates additional padding at finer scales)
        p.starty(f) = ay-virtpady;
        p.startx(f) = ax-virtpadx;      
        p.step   = step;
      end
      components{c}(k) = p;
    end
  end
  
  resp    = cell(length(pyra.feat),1);
  filters = cell(length(model.filters),1);
  for i = 1:length(filters),
    filters{i} = model.filters(i).w;
  end

function [score,Ix,Iy,Ik] = passmsg(child,parent)
  K   = length(child.filterid);
  Ny  = size(parent.score,1);
  Nx  = size(parent.score,2);  
  [Ix0,Iy0,score0] = deal(zeros([Ny Nx K]));

  for k = 1:K
    [score0(:,:,k),Ix0(:,:,k),Iy0(:,:,k)] = shiftdt(child.score(:,:,k), child.w(1,k), child.w(2,k), child.w(3,k), child.w(4,k),child.startx(k),child.starty(k),Nx,Ny,child.step);
  end

  % At each parent location, for each parent mixture 1:L, compute best child mixture 1:K
  L  = length(parent.filterid);
  N  = Nx*Ny;
  i0 = reshape(1:N,Ny,Nx);
  [score,Ix,Iy,Ix,Ik] = deal(zeros(Ny,Nx,L));
  for l = 1:L
    b = child.b(1,l,:);
    [score(:,:,l),I] = max(bsxfun(@plus,score0,b),[],3);
    i = i0 + N*(I-1);
    Ix(:,:,l)    = Ix0(i);
    Iy(:,:,l)    = Iy0(i);
    Ik(:,:,l)    = I;
  end
  
function [score,Ix,Iy,Ik] = passmsg_d(parent,child)
  K  = length(child.filterid);
  L  = length(parent.filterid);
  Ny = size(child.score,1);
  Nx = size(child.score,2);  
  N  = size(parent.score,1);
  [Ix,Iy,Ik,score] = deal(zeros([Ny Nx K]));
  score0 = parent.mm - child.msg;

  for k = 1:K
    [tmp Ik0] = max(bsxfun(@plus,score0,reshape(child.b(1,:,k),[1 1 L])),[],3);
    [score(:,:,k),Ix(:,:,k),Iy(:,:,k)] = shiftdt(tmp, child.w(1,k), -child.w(2,k), child.w(3,k), -child.w(4,k),2-child.startx(k),2-child.starty(k),Nx,Ny,child.step);
    Ik(:,:,k) = Ik0((Ix(:,:,k)-1)*N+Iy(:,:,k));
  end


  
