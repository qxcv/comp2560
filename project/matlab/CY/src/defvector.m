% Compute the deformation feature given parent locations,
% child locations, and the child part
function res = defvector(part, x1, y1, x2, y2, m, id)
probx = x1 - part.mean_x{id}(m);
proby = y1 - part.mean_y{id}(m);

var_x = part.var_x{id}(m);
var_y = part.var_y{id}(m);

dx = (probx - x2) / var_x;
dy = (proby - y2) / var_y;
res = -[dx^2, dx, dy^2, dy]'; % gaussian normalization + variances
