// DAY: 49
// TITLE_ZH: PCA、SVD 與 t-SNE：把幾百欄資料壓成兩欄，畫成一張看得懂的圖
// TITLE_EN: PCA, SVD and t-SNE: turning hundreds of columns into two you can plot
// SUB_ZH: 降維把 d 欄壓成兩欄，而被壓掉的那些不會留下任何痕跡——出來的是一張圖，而圖極容易被過度解讀。PCA 回答「哪些方向的變異最大」，不是「哪些方向有用」；t-SNE 回答「誰跟誰是鄰居」，不是「這兩群離多遠、誰比較大」。忘了減平均會讓第一主成分指向資料所在的位置而不是它的形狀，而且還報出更漂亮的分數；98% 的變異可能一點訊號都沒有；共變異數矩陣會把真實的方向直接四捨五入成 0；t-SNE 會在純雜訊裡畫出三群，而且換個隨機種子就換一組。
// SUB_EN: Dimensionality reduction takes d columns and returns two, and the ones it drops leave no trace. What comes out is a picture, and a picture is very easy to over-read. PCA answers which directions carry the most variance, not which ones matter; t-SNE answers who is whose neighbour, not how far apart the groups are or how big they are. Forgetting to subtract the mean makes the first component point at where the data sits rather than at its shape - and report a better score for it; 98% of the variance can hold none of the signal; the covariance matrix rounds a real direction out of existence; and t-SNE draws three clusters in pure noise, then draws different ones when the seed changes.
// FOLDER: day%2049%20-%20pca%20svd%20and%20t-sne
// MEDIUM: https://medium.com/100-days-of-python
const VIEW = [9.8, 6.4];
function chip(t, cls){ return {t:t, cls:cls || ''}; }
const curV = () => (typeof varIx !== 'undefined' ? varIx : 0);

const isCJK = ch => { const c = ch.codePointAt(0);
  return (c >= 0x2e80 && c <= 0x9fff) || (c >= 0xff00 && c <= 0xff60) || (c >= 0x3000 && c <= 0x303f); };
function tw(str, fs){ let w = 0;
  for (const ch of String(str)) w += isCJK(ch) ? 1.0 : .55;
  return w * fs; }
function fitT(x, y, s, opt){
  opt = Object.assign({}, opt || {});
  const fs = opt.fs || .32, a = opt.anchor || 'middle', pad = .20;
  const room = a === 'start' ? VIEW[0] - pad - x
             : a === 'end'   ? x - pad
             : 2 * Math.min(x - pad, VIEW[0] - pad - x);
  const strs = (typeof s === 'string') ? [s] : [s.zh, s.en].filter(t => t != null);
  const w = Math.max.apply(null, strs.map(t => tw(t, fs)));
  if (w > room) opt.fs = fs * room / w;
  return S.t(x, y, s, opt);
}
function fitFs(lab, w, fs){
  const strs = (typeof lab === 'string') ? [lab] : [lab.zh, lab.en];
  const need = Math.max.apply(null, strs.map(t => tw(t, fs)));
  return need > w - .08 ? fs * (w - .08) / need : fs;
}
const box = (x, y, w, h, s, lab, fs) => S.r(x, y, w, h, s, lab, {fs:fitFs(lab, w, fs || h * .5)});
function caps(sh, o){
  if (o.cap) sh.push(fitT(4.9, 5.62, o.cap, {c:o.capc || COL.pale, fs:.30}));
  if (o.cap2) sh.push(fitT(4.9, 6.06, o.cap2, {c:COL.grey, fs:.26}));
  return sh;
}
const side = (sh, x, y0, lines, dy) => (lines || []).forEach((l, k) =>
  sh.push(fitT(x, y0 + k * (dy || .5), l[0], {anchor:'start', c:l[1] || COL.pale, fs:l[2] || .30})));
const fmtN = x => !isFinite(x) ? (Number.isNaN(x) ? 'nan' : (x > 0 ? 'inf' : '-inf'))
  : String(Math.round(x * 1e3) / 1e3);

/* ================================================================ shared geometry */
// A plot box in view coordinates.  Data y grows upward, the SVG y grows downward,
// so every mapping flips it once - exactly once.
function Plot(x0, y0, w, h, lim){
  this.x0 = x0; this.y0 = y0; this.w = w; this.h = h; this.lim = lim;
}
Plot.prototype.px = function(p){
  const [[ax, ay], [bx, by]] = this.lim, pad = .07;
  const ux = (p[0] - ax) / (bx - ax), uy = (p[1] - ay) / (by - ay);
  return [this.x0 + this.w * (pad + ux * (1 - 2 * pad)),
          this.y0 + this.h * (1 - pad - uy * (1 - 2 * pad))];
};
Plot.prototype.frame = function(sh, st){
  sh.push(S.r(this.x0, this.y0, this.w, this.h, st || 'ghost', null));
  return sh;
};
function dots(sh, plot, pts, styles, labs, r){
  pts.forEach((p, i) => {
    const q = plot.px(p);
    sh.push(S.c(q[0], q[1], r || .12, styles[i] || 'idle', labs ? labs[i] : null,
                labs ? {fs:.20} : {}));
  });
  return sh;
}
function crosses(sh, plot, pts, styles, labs, ghostAt){
  pts.forEach((p, i) => {
    const st = styles[i] || 'hot';
    if (!isFinite(p[0]) || !isFinite(p[1])){
      // nan has no position: draw it where it used to be, as a box that says so.
      const g = plot.px((ghostAt && ghostAt[i]) || [0, 0]);
      sh.push(box(g[0] - .42, g[1] - .22, .84, .44, 'bad', 'nan', .26));
      if (labs && labs[i]) sh.push(fitT(g[0], g[1] - .46, labs[i], {c:COL.red, fs:.22}));
      return;
    }
    const q = plot.px(p), d = .17;
    sh.push(S.e(q[0] - d, q[1] - d, q[0] + d, q[1] + d, {s:st, arrow:false, w:.05}));
    sh.push(S.e(q[0] - d, q[1] + d, q[0] + d, q[1] - d, {s:st, arrow:false, w:.05}));
    if (labs && labs[i]) sh.push(fitT(q[0] + .30, q[1] - .02, labs[i],
                                      {anchor:'start', c:COL.orange, fs:.24}));
  });
  return sh;
}
function bars(sh, x0, yBase, bw, gap, vals, styles, labs, maxH, top){
  const hi = top || Math.max.apply(null, vals.map(Math.abs));
  vals.forEach((v, i) => {
    const h = Math.max(.04, maxH * Math.abs(v) / hi), x = x0 + i * (bw + gap);
    sh.push(S.r(x, yBase - h, bw, h, styles[i] || 'idle', null));
    if (labs && labs[i] != null)
      sh.push(fitT(x + bw / 2, h > .55 ? yBase - h + .26 : yBase - h - .22, labs[i],
                   {c:COL.pale, fs:.24}));
  });
  return sh;
}

/* ================================================================ data from dimred.py */
const PTS = [[8.185, 4.984], [5.142, 3.938], [7.195, 4.903], [11.082, 6.142], [9.208, 5.633], [8.446, 5.436], [5.760, 4.739], [4.617, 3.488], [7.950, 5.083], [5.284, 4.444], [8.891, 5.424], [7.234, 4.103], [11.753, 6.374], [9.126, 5.186], [5.142, 4.007], [9.427, 6.454], [7.680, 5.566], [1.216, 2.761], [9.672, 4.980], [6.686, 4.583], [8.716, 5.001], [6.363, 4.858], [8.317, 5.275], [5.946, 4.630], [8.104, 4.517], [5.955, 5.452], [5.878, 4.009], [9.035, 5.811], [7.295, 4.548], [6.333, 5.130], [5.713, 5.086], [8.232, 5.571], [9.117, 5.271], [6.475, 5.163], [7.086, 4.652], [6.490, 4.690], [10.614, 5.795], [4.503, 4.059], [8.245, 5.923], [8.017, 5.063], [6.107, 4.347], [7.075, 5.371], [6.262, 4.873]];
const PC_CENTRED = [-0.949765, -0.312964];
const PC_UNCENTRED = [-0.835584, -0.549363];
const PC_MEAN = [7.3389, 4.9610];
const CENTRE_NUM = {evrC:0.9653, evrU:0.9948, angPC:15.1, angMean:0.7};
const HID = [[-2.582, -2.094], [-4.269, -2.861], [2.099, 0.094], [-6.822, -5.142], [-0.312, 0.675], [-1.611, -0.133], [8.555, 5.778], [-3.909, -3.373], [9.316, 5.973], [-10.136, -5.044], [-1.901, -2.429], [2.477, 0.534], [-4.414, -1.102], [-2.416, -2.421], [-5.990, -2.464], [-6.935, -5.179], [-0.742, -1.601], [-10.442, -6.941], [6.325, 2.987], [-2.266, -1.094], [-2.438, -2.083], [-4.594, -3.329], [1.835, -0.127], [4.137, 2.910], [2.799, 0.734], [-3.300, -3.055], [-3.178, -1.082], [-2.385, -2.261], [-11.636, -7.548], [7.789, 5.238], [-6.216, -3.070], [4.549, 3.527], [-3.686, -3.116], [5.559, 2.135], [0.967, 1.619], [0.934, 1.603], [-0.393, -1.468], [4.524, 3.594], [-6.276, -4.707], [5.723, 4.472], [1.286, 0.058], [-3.395, -0.842], [-4.772, -1.820], [-5.695, -2.232], [-3.493, -3.095], [-1.582, -1.688], [-1.812, -0.088], [-3.805, -1.534], [-0.976, 0.527], [-0.198, 0.777], [-6.557, -2.807], [-3.853, -1.697], [5.192, 1.612], [2.234, 2.555], [-2.787, -1.031], [-4.571, -3.313], [-0.993, 0.310], [5.651, 2.428], [1.500, 0.025], [2.303, 0.909], [1.549, 0.140], [-6.758, -3.139], [3.375, 1.276], [0.314, -0.487], [4.959, 1.950], [-0.094, 0.926], [19.673, 10.945], [4.231, 1.398], [-0.241, 0.727], [-5.617, -2.322], [0.866, 1.471], [-7.404, -3.420], [-1.832, 0.603], [-4.422, -1.668], [-2.370, -0.124], [0.586, -0.423], [-6.168, -3.033], [5.512, 3.760], [-0.863, 0.595], [-2.296, 0.081], [-7.739, -3.311], [0.998, -0.075], [-2.826, -0.046], [1.498, -0.486], [-2.562, -0.722], [-0.008, 1.083], [3.671, 3.049], [4.529, 1.666], [4.925, 1.957], [1.435, 1.530]];
const HID_Y = [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1];
const HID_PC = [[0.867705, 0.497079], [0.497079, -0.867705]];
const HID_MEAN = [-0.3420, -0.1818];
const HID_NUM = {evr:[0.9833, 0.0167], acc:[0.5450, 1.0000], ratio:59.0};
const COND = [{eps:0.001, svd:7.042477e-03, eig:7.042477e-03, cond:1.999e+03, ccond:3.997e+06}, {eps:0.0001, svd:7.042477e-04, eig:7.042476e-04, cond:1.999e+04, ccond:3.997e+08}, {eps:1e-05, svd:7.042477e-05, eig:7.042471e-05, cond:1.999e+05, ccond:3.997e+10}, {eps:1e-06, svd:7.042477e-06, eig:7.041634e-06, cond:1.999e+06, ccond:3.997e+12}, {eps:1e-07, svd:7.042477e-07, eig:7.402139e-07, cond:1.999e+07, ccond:3.975e+14}, {eps:1e-08, svd:7.042477e-08, eig:0.000000e+00, cond:1.999e+08, ccond:4.171e+16}, {eps:1e-09, svd:7.042477e-09, eig:0.000000e+00, cond:1.999e+09, ccond:6.472e+17}];
/* ccond is measured from the covariance matrix, not cond^2 - at the bottom end the
   matrix is too degenerate for its own condition number to survive, which is why the
   last row reads 6.5e+17 instead of the predicted 4.0e+18. */
const MAT = [[1.047, 0.804, 0.644, -0.149, 0.566, 0.041, -0.655, -0.634, -0.903, -1.130, -0.636, -0.731], [1.187, 1.006, 1.130, 0.706, 0.522, -0.127, -0.273, -0.781, -0.493, -0.675, -0.528, -0.282], [1.266, 1.649, 1.780, 0.571, 0.202, -0.102, 0.155, -0.277, -0.185, -0.292, -0.309, -0.064], [1.687, 1.882, 1.194, 1.086, 0.907, 0.935, -0.035, -0.367, -0.384, -0.019, -0.209, 0.407], [1.419, 2.104, 1.893, 0.994, 1.041, 0.946, 0.335, 0.309, 0.508, -0.036, -0.063, 0.041], [2.141, 1.864, 1.681, 1.414, 1.257, 0.467, 0.022, -0.457, 0.305, 0.006, 0.476, 0.323], [1.812, 2.052, 1.726, 1.146, 0.893, 1.195, 0.513, 0.274, -0.044, -0.166, 0.341, 0.318], [1.754, 1.844, 1.464, 1.453, 1.342, 0.489, 0.727, -0.051, 0.009, -0.257, 0.008, 0.302], [1.710, 1.578, 1.397, 1.075, 0.543, 0.508, 0.346, 0.219, 0.008, 0.442, 0.019, 0.051], [2.102, 1.307, 1.623, 0.857, 0.839, -0.103, 0.285, -0.233, -0.581, 0.062, -0.054, -0.008], [1.216, 1.326, 1.109, 1.045, 0.718, -0.009, -0.308, -0.558, -0.908, -0.737, -0.501, -0.078], [1.471, 0.874, 1.026, 0.492, 0.776, -0.116, -0.307, -0.920, -1.035, -0.800, -0.835, -0.424]];
const MAT_SV = [10.3624, 3.4556, 1.3221, 1.2479, 0.9166, 0.8417, 0.7851, 0.5502, 0.3998, 0.2886, 0.2470, 0.0106];
const MAT_REC = {'1':{m:[[0.840, 0.831, 0.749, 0.502, 0.438, 0.214, 0.070, -0.120, -0.118, -0.117, -0.061, 0.028], [1.167, 1.155, 1.041, 0.698, 0.608, 0.297, 0.098, -0.166, -0.163, -0.163, -0.085, 0.039], [1.424, 1.410, 1.271, 0.852, 0.742, 0.363, 0.119, -0.203, -0.199, -0.199, -0.103, 0.048], [1.725, 1.707, 1.539, 1.032, 0.899, 0.440, 0.144, -0.246, -0.241, -0.241, -0.125, 0.058], [1.832, 1.814, 1.635, 1.096, 0.955, 0.467, 0.153, -0.261, -0.257, -0.256, -0.133, 0.062], [2.005, 1.985, 1.789, 1.200, 1.045, 0.511, 0.168, -0.286, -0.281, -0.280, -0.146, 0.067], [1.935, 1.916, 1.727, 1.158, 1.009, 0.493, 0.162, -0.276, -0.271, -0.271, -0.141, 0.065], [1.888, 1.869, 1.685, 1.130, 0.984, 0.481, 0.158, -0.269, -0.264, -0.264, -0.137, 0.063], [1.555, 1.539, 1.387, 0.930, 0.810, 0.396, 0.130, -0.222, -0.218, -0.217, -0.113, 0.052], [1.665, 1.648, 1.486, 0.996, 0.868, 0.424, 0.139, -0.237, -0.233, -0.233, -0.121, 0.056], [1.364, 1.350, 1.217, 0.816, 0.711, 0.348, 0.114, -0.194, -0.191, -0.191, -0.099, 0.046], [1.222, 1.209, 1.090, 0.731, 0.637, 0.311, 0.102, -0.174, -0.171, -0.171, -0.089, 0.041]], err:4.2448, pred:4.2448, kept:0.8563}, '2':{m:[[1.013, 0.645, 0.725, 0.232, 0.456, -0.312, -0.503, -0.844, -0.997, -0.892, -0.730, -0.476], [1.273, 1.041, 1.026, 0.532, 0.619, -0.026, -0.254, -0.611, -0.703, -0.638, -0.495, -0.270], [1.451, 1.381, 1.267, 0.811, 0.745, 0.282, 0.031, -0.314, -0.334, -0.318, -0.206, -0.029], [1.710, 1.723, 1.541, 1.055, 0.897, 0.484, 0.193, -0.185, -0.167, -0.176, -0.069, 0.100], [1.741, 1.911, 1.648, 1.238, 0.945, 0.743, 0.453, 0.118, 0.204, 0.149, 0.217, 0.326], [1.951, 2.044, 1.797, 1.285, 1.039, 0.677, 0.349, -0.057, -0.003, -0.036, 0.066, 0.227], [1.845, 2.013, 1.740, 1.299, 0.999, 0.768, 0.461, 0.102, 0.188, 0.133, 0.209, 0.328], [1.834, 1.927, 1.692, 1.214, 0.979, 0.646, 0.337, -0.043, 0.011, -0.022, 0.072, 0.221], [1.486, 1.613, 1.397, 1.037, 0.803, 0.605, 0.357, 0.065, 0.130, 0.089, 0.152, 0.252], [1.688, 1.625, 1.483, 0.962, 0.870, 0.357, 0.066, -0.330, -0.346, -0.332, -0.207, -0.009], [1.468, 1.239, 1.203, 0.654, 0.722, 0.033, -0.228, -0.627, -0.716, -0.653, -0.499, -0.255], [1.386, 1.034, 1.067, 0.475, 0.654, -0.185, -0.439, -0.858, -1.001, -0.901, -0.720, -0.434]], err:2.4652, pred:2.4652, kept:0.9515}, '3':{m:[[0.818, 0.829, 0.690, 0.110, 0.490, 0.045, -0.599, -0.750, -0.955, -1.121, -0.823, -0.540], [1.304, 1.012, 1.032, 0.551, 0.614, -0.081, -0.239, -0.625, -0.709, -0.603, -0.481, -0.260], [1.487, 1.348, 1.274, 0.833, 0.739, 0.217, 0.049, -0.331, -0.342, -0.276, -0.189, -0.018], [1.609, 1.818, 1.523, 0.992, 0.915, 0.669, 0.143, -0.136, -0.145, -0.294, -0.117, 0.067], [1.531, 2.108, 1.610, 1.108, 0.982, 1.127, 0.351, 0.219, 0.249, -0.097, 0.117, 0.256], [2.079, 1.924, 1.820, 1.364, 1.018, 0.444, 0.411, -0.119, -0.031, 0.114, 0.127, 0.269], [1.689, 2.159, 1.712, 1.203, 1.026, 1.052, 0.385, 0.177, 0.221, -0.049, 0.134, 0.277], [1.865, 1.898, 1.698, 1.234, 0.973, 0.589, 0.352, -0.058, 0.004, 0.015, 0.087, 0.231], [1.579, 1.526, 1.413, 1.095, 0.787, 0.435, 0.402, 0.020, 0.110, 0.198, 0.196, 0.282], [1.962, 1.368, 1.532, 1.132, 0.823, -0.144, 0.200, -0.462, -0.405, -0.011, -0.076, 0.082], [1.455, 1.251, 1.200, 0.646, 0.724, 0.057, -0.235, -0.621, -0.713, -0.668, -0.505, -0.259], [1.419, 1.003, 1.073, 0.496, 0.649, -0.245, -0.423, -0.873, -1.008, -0.863, -0.704, -0.424]], err:2.0807, pred:2.0807, kept:0.9655}, '5':{m:[[0.851, 0.833, 0.703, 0.065, 0.440, 0.069, -0.579, -0.699, -1.017, -1.079, -0.843, -0.558], [1.193, 1.039, 1.116, 0.597, 0.649, -0.203, -0.263, -0.705, -0.517, -0.700, -0.464, -0.274], [1.338, 1.512, 1.796, 0.562, 0.366, -0.087, 0.144, -0.162, -0.112, -0.283, -0.324, -0.268], [1.762, 1.719, 1.210, 1.088, 1.068, 0.905, 0.115, -0.158, -0.398, -0.219, -0.065, 0.197], [1.412, 2.189, 1.865, 1.023, 0.850, 0.940, 0.376, 0.245, 0.444, -0.152, 0.071, 0.148], [1.966, 1.875, 1.663, 1.608, 1.301, 0.402, 0.312, -0.365, 0.183, -0.059, 0.237, 0.391], [1.793, 2.138, 1.648, 1.147, 0.978, 1.162, 0.412, 0.263, 0.039, 0.047, 0.113, 0.282], [1.825, 1.851, 1.549, 1.397, 1.171, 0.605, 0.288, -0.210, 0.088, -0.076, 0.163, 0.328], [1.679, 1.548, 1.485, 0.932, 0.603, 0.496, 0.470, 0.194, -0.076, 0.331, 0.123, 0.212], [2.059, 1.382, 1.578, 0.993, 0.669, -0.078, 0.258, -0.310, -0.583, 0.111, -0.137, 0.026], [1.448, 1.218, 1.094, 0.739, 0.840, 0.087, -0.271, -0.701, -0.694, -0.708, -0.461, -0.197], [1.466, 0.976, 0.988, 0.517, 0.685, -0.176, -0.428, -0.873, -1.087, -0.836, -0.692, -0.389]], err:1.3899, pred:1.3899, kept:0.9846}, '12':{m:[[1.047, 0.804, 0.644, -0.149, 0.566, 0.041, -0.655, -0.634, -0.903, -1.130, -0.636, -0.731], [1.187, 1.006, 1.130, 0.706, 0.522, -0.127, -0.273, -0.781, -0.493, -0.675, -0.528, -0.282], [1.266, 1.649, 1.780, 0.571, 0.202, -0.102, 0.155, -0.277, -0.185, -0.292, -0.309, -0.064], [1.687, 1.882, 1.194, 1.086, 0.907, 0.935, -0.035, -0.367, -0.384, -0.019, -0.209, 0.407], [1.419, 2.104, 1.893, 0.994, 1.041, 0.946, 0.335, 0.309, 0.508, -0.036, -0.063, 0.041], [2.141, 1.864, 1.681, 1.414, 1.257, 0.467, 0.022, -0.457, 0.305, 0.006, 0.476, 0.323], [1.812, 2.052, 1.726, 1.146, 0.893, 1.195, 0.513, 0.274, -0.044, -0.166, 0.341, 0.318], [1.754, 1.844, 1.464, 1.453, 1.342, 0.489, 0.727, -0.051, 0.009, -0.257, 0.008, 0.302], [1.710, 1.578, 1.397, 1.075, 0.543, 0.508, 0.346, 0.219, 0.008, 0.442, 0.019, 0.051], [2.102, 1.307, 1.623, 0.857, 0.839, -0.103, 0.285, -0.233, -0.581, 0.062, -0.054, -0.008], [1.216, 1.326, 1.109, 1.045, 0.718, -0.009, -0.308, -0.558, -0.908, -0.737, -0.501, -0.078], [1.471, 0.874, 1.026, 0.492, 0.776, -0.116, -0.307, -0.920, -1.035, -0.800, -0.835, -0.424]], err:0.0000, pred:0.0000, kept:1.0000}};
const TSNE_SNAPS = [[[-0.001, 0.002], [-0.005, 0.001], [0.002, -0.001], [-0.009, -0.005], [0.001, 0.003], [0.002, 0.001], [0.005, 0.001], [0.004, 0.004], [0.002, 0.003], [-0.004, -0.004], [0.000, -0.006], [0.002, -0.001], [-0.006, 0.001], [0.003, 0.005], [0.000, -0.000], [0.003, 0.002], [-0.000, -0.002], [-0.003, -0.001], [0.003, 0.002], [-0.004, -0.005], [0.008, -0.005], [-0.004, -0.003], [-0.000, 0.003], [-0.005, -0.011], [-0.003, -0.003], [0.000, 0.005], [0.001, -0.003], [0.008, -0.002], [-0.001, -0.001], [0.008, 0.003], [0.005, 0.006], [-0.004, 0.002], [0.001, 0.003], [-0.005, -0.004], [-0.003, 0.011], [0.003, -0.002], [-0.002, 0.004], [-0.007, 0.007], [0.004, -0.002], [0.000, -0.008], [-0.001, 0.002], [0.001, 0.005], [0.007, -0.003], [-0.004, -0.006], [0.004, -0.008], [0.000, -0.007], [0.002, 0.003], [-0.002, -0.004], [-0.001, 0.002], [0.006, 0.008], [-0.004, -0.005], [0.000, 0.008], [-0.004, 0.006], [0.003, -0.001], [0.005, -0.000], [0.003, -0.000], [-0.001, 0.001], [-0.004, 0.004], [-0.007, -0.002], [-0.004, -0.004]], [[4.270, -1.886], [4.973, -5.441], [6.130, -0.048], [0.705, 4.517], [4.561, -0.787], [3.296, -3.606], [5.277, -1.898], [2.862, -1.385], [4.415, 3.047], [7.475, -0.249], [5.198, 0.395], [4.482, -1.644], [3.795, -2.990], [4.632, -2.956], [0.574, -3.352], [3.460, -2.386], [4.661, -1.333], [4.252, 1.933], [6.286, -8.125], [5.167, -1.199], [-0.269, -0.553], [-3.079, -3.077], [-2.843, 3.684], [-3.214, -4.154], [-2.282, 0.555], [-5.862, -1.634], [-10.263, -1.437], [-5.313, 5.471], [-5.511, -0.046], [-1.443, -1.109], [-0.837, -0.811], [-2.667, -0.769], [-3.319, -1.660], [-3.062, -1.110], [-2.964, -2.141], [-5.726, -3.029], [-3.067, 0.302], [-6.225, -0.368], [-1.158, 0.139], [-2.324, 3.370], [-0.844, -0.280], [-1.715, 2.879], [0.434, 1.424], [-0.014, 11.104], [3.884, 4.559], [-1.964, -1.181], [-1.179, 0.593], [-3.346, 0.965], [-0.850, 2.529], [-0.957, 0.145], [-0.201, 0.850], [-1.143, 2.304], [0.068, 1.534], [-1.184, 1.382], [-0.801, 4.368], [-2.778, 0.608], [-2.045, 1.181], [-0.602, -0.558], [-0.658, 2.248], [0.851, 1.116]], [[12.016, 1.022], [-1.567, 0.275], [5.926, -3.302], [5.190, 0.022], [4.774, -2.289], [5.646, -7.490], [9.231, -1.148], [6.113, -1.757], [6.010, -2.027], [3.762, -2.184], [5.406, -2.144], [5.832, -2.468], [6.436, -2.425], [3.328, -8.264], [5.758, -2.946], [5.120, -2.305], [5.688, -0.846], [6.060, -2.244], [4.019, 1.047], [5.583, 1.098], [-6.434, -2.060], [-10.214, -4.549], [-5.758, 2.191], [-3.929, -2.921], [-8.003, -1.724], [-5.261, 1.891], [-2.877, 0.017], [-7.390, -3.365], [-8.058, -1.951], [-5.871, -2.400], [-2.378, -3.506], [-7.272, -2.409], [-5.224, -3.044], [-6.396, -1.620], [-7.627, -1.828], [-4.104, -2.130], [-3.542, -2.459], [-6.669, 0.827], [-5.625, -0.926], [-0.750, -1.882], [-1.224, 5.267], [10.937, 3.949], [1.892, 0.034], [6.506, 4.053], [-1.219, 5.080], [-1.761, 5.290], [-1.070, 5.789], [-1.224, 3.848], [-1.394, 2.175], [0.402, 4.156], [-0.540, 4.269], [-1.231, 3.991], [-1.907, 4.868], [6.497, 6.209], [-1.799, 4.396], [0.511, 4.321], [-2.948, -3.421], [-4.022, 4.060], [-0.984, 1.853], [-2.369, 4.037]], [[9.480, -3.535], [8.860, -2.824], [8.371, -4.908], [6.119, -3.421], [5.738, -3.516], [-5.783, -3.182], [6.164, -4.492], [10.435, -0.658], [7.436, -4.872], [6.868, -4.359], [9.235, -3.983], [7.406, -1.693], [-0.014, -1.241], [-0.953, -1.517], [9.335, -4.400], [6.840, -3.400], [8.474, -1.168], [11.897, -5.197], [15.253, -5.379], [7.595, -4.018], [-7.680, -2.371], [-8.122, -1.680], [-7.306, -1.999], [-7.686, -2.703], [-8.456, -1.655], [-6.353, -2.302], [-7.709, -0.099], [-7.270, -0.623], [-12.871, -2.709], [-6.405, -2.426], [-7.099, -2.325], [-5.842, -2.261], [-6.999, -2.129], [-6.660, -1.797], [-8.386, -11.242], [-1.625, -2.870], [-6.469, -1.733], [-5.906, -1.738], [-6.687, -2.534], [-6.560, 4.028], [1.000, 3.960], [-0.525, 3.179], [9.014, 5.263], [2.705, 5.530], [2.570, 9.182], [0.512, 8.526], [0.896, 4.857], [-0.436, 6.658], [-0.071, 6.690], [-1.665, 6.098], [0.076, 4.941], [1.881, 0.511], [-2.757, 7.000], [0.038, 6.710], [-1.490, 4.504], [-2.439, 4.630], [-0.668, 5.905], [-2.543, 4.556], [-1.939, 6.578], [-0.824, 5.652]], [[9.244, -5.166], [9.807, -4.522], [8.648, -4.345], [8.887, -4.439], [7.479, -4.119], [8.176, -5.620], [10.299, -3.432], [9.186, -3.582], [8.456, -5.497], [9.623, -4.187], [9.766, -4.937], [8.426, -5.307], [8.671, -4.479], [8.978, -5.500], [9.668, -3.688], [9.017, -3.633], [8.686, -3.816], [9.787, -4.221], [9.547, -4.479], [8.357, -5.850], [-10.247, -3.013], [-9.452, -2.420], [-9.958, -3.237], [-8.931, -4.089], [-8.676, -2.178], [-9.191, -4.352], [-9.934, -3.432], [-9.649, -3.714], [-10.509, -3.533], [-9.781, -3.898], [-8.838, -3.545], [-8.382, -4.047], [-9.143, -2.992], [-10.518, -4.231], [-9.856, -3.929], [-9.074, -2.614], [-9.695, -2.149], [-9.110, -4.441], [-9.734, -4.997], [-10.497, -3.751], [-0.933, 7.997], [1.790, 8.125], [0.770, 8.181], [0.560, 8.067], [-0.026, 7.987], [1.423, 8.144], [0.394, 9.130], [1.525, 7.802], [0.887, 9.263], [1.328, 8.015], [0.699, 8.584], [0.768, 7.581], [-0.227, 8.760], [0.136, 7.314], [-0.425, 6.642], [0.352, 9.140], [0.610, 7.384], [0.241, 8.736], [-0.118, 7.598], [0.712, 6.932]], [[10.462, -6.085], [10.923, -5.281], [9.740, -5.264], [10.014, -5.358], [8.564, -4.921], [9.206, -6.397], [11.238, -4.100], [10.170, -4.444], [9.528, -6.364], [10.685, -4.989], [10.930, -5.730], [9.710, -6.358], [9.787, -5.436], [10.161, -6.431], [10.663, -4.474], [10.009, -4.512], [9.725, -4.792], [10.860, -4.984], [10.676, -5.339], [9.458, -6.756], [-11.460, -3.782], [-10.719, -3.099], [-11.139, -3.975], [-10.003, -4.730], [-9.952, -2.769], [-10.243, -5.026], [-11.092, -4.169], [-10.768, -4.425], [-11.672, -4.335], [-10.885, -4.631], [-9.970, -4.166], [-9.457, -4.622], [-10.338, -3.643], [-11.602, -5.051], [-10.957, -4.671], [-10.310, -3.254], [-10.996, -2.850], [-10.151, -5.107], [-10.734, -5.736], [-11.633, -4.558], [-0.896, 9.543], [1.870, 9.724], [0.831, 9.750], [0.609, 9.630], [0.021, 9.532], [1.493, 9.730], [0.423, 10.704], [1.609, 9.381], [0.928, 10.847], [1.398, 9.596], [0.747, 10.155], [0.840, 9.137], [-0.208, 10.313], [0.200, 8.846], [-0.376, 8.174], [0.375, 10.715], [0.695, 8.926], [0.278, 10.294], [-0.065, 9.139], [0.801, 8.474]], [[13.245, -7.892], [13.590, -6.991], [12.378, -7.182], [12.672, -7.232], [11.130, -7.027], [12.015, -8.400], [13.691, -5.750], [12.664, -6.275], [12.343, -8.318], [13.283, -6.735], [13.669, -7.448], [12.544, -8.289], [12.455, -7.350], [13.010, -8.284], [13.177, -6.221], [12.520, -6.371], [12.275, -6.702], [13.469, -6.703], [13.335, -7.099], [12.345, -8.732], [-14.277, -5.404], [-13.639, -4.593], [-13.909, -5.546], [-12.634, -6.115], [-12.917, -4.134], [-12.826, -6.461], [-13.831, -5.736], [-13.455, -5.939], [-14.395, -6.001], [-13.541, -6.176], [-12.699, -5.539], [-12.100, -5.904], [-13.158, -5.079], [-14.196, -6.724], [-13.608, -6.229], [-13.192, -4.682], [-13.963, -4.392], [-12.717, -6.527], [-13.204, -7.260], [-14.310, -6.225], [-0.874, 12.888], [1.966, 13.075], [0.898, 13.095], [0.665, 12.971], [0.064, 12.880], [1.575, 13.083], [0.481, 14.089], [1.693, 12.707], [1.009, 14.230], [1.474, 12.944], [0.816, 13.527], [0.904, 12.487], [-0.172, 13.689], [0.243, 12.194], [-0.340, 11.494], [0.432, 14.098], [0.761, 12.263], [0.326, 13.672], [-0.027, 12.485], [0.864, 11.795]], [[15.475, -9.121], [15.635, -8.159], [14.470, -8.595], [14.775, -8.585], [13.208, -8.691], [14.349, -9.867], [15.481, -6.909], [14.571, -7.638], [14.659, -9.721], [15.274, -7.968], [15.806, -8.595], [14.863, -9.655], [14.580, -8.745], [15.325, -9.558], [15.066, -7.481], [14.445, -7.763], [14.269, -8.144], [15.455, -7.897], [15.402, -8.321], [14.748, -10.134], [-16.451, -6.625], [-15.884, -5.755], [-16.067, -6.736], [-14.733, -7.190], [-15.196, -5.226], [-14.894, -7.559], [-15.972, -6.917], [-15.574, -7.088], [-16.519, -7.234], [-15.640, -7.336], [-14.852, -6.616], [-14.209, -6.926], [-15.354, -6.197], [-16.248, -7.948], [-15.705, -7.399], [-15.425, -5.804], [-16.228, -5.581], [-14.776, -7.615], [-15.204, -8.397], [-16.411, -7.456], [-0.851, 15.346], [2.017, 15.569], [0.935, 15.581], [0.698, 15.457], [0.094, 15.353], [1.619, 15.576], [0.502, 16.568], [1.745, 15.187], [1.036, 16.720], [1.518, 15.431], [0.848, 16.007], [0.949, 14.953], [-0.155, 16.158], [0.284, 14.647], [-0.292, 13.942], [0.452, 16.582], [0.811, 14.726], [0.349, 16.141], [0.007, 14.945], [0.920, 14.260]]];
const TSNE_ITERS = [1, 25, 50, 100, 150, 250, 400, 600];
const TSNE_KL = [1.3166, 1.5288, 1.1877, 1.1497, 0.0569, 0.0521, 0.0463, 0.0440];
const TSNE_Y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2];
const TSNE_RECALL = 0.8617;
const READ = {oSpread:[1.025, 10.818, 1.182], eSpread:[2.159, 2.421, 2.155], dist:[[2.02, 1.04], [4.03, 1.04], [8.05, 1.04], [26.82, 1.02]], silRaw:0.0872, silEmb:[0.3972, 0.3763, 0.3707, 0.3689], corBlobs:0.9889, corNoise:0.3186};
const LC719 = {nums:[1, 4, 4, 6, 9, 12], k:5, steps:[{lo:0, hi:11, mid:5, cnt:10, go:true}, {lo:0, hi:5, mid:2, cnt:3, go:false}, {lo:3, hi:5, mid:4, cnt:7, go:true}, {lo:3, hi:4, mid:3, cnt:7, go:true}, {lo:3, hi:3, mid:null, cnt:null, go:null}], answer:3, all:[0, 2, 2, 3, 3, 3, 3, 5, 5, 5, 6, 8, 8, 8, 11]};

/* ================================================================ shared helpers */
const PURPLE = '#b45cff';
function equalLim(pts, w, h, pad){
  pad = pad == null ? .08 : pad;
  let lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
  pts.forEach(p => { for (let j = 0; j < 2; j++){ lo[j] = Math.min(lo[j], p[j]); hi[j] = Math.max(hi[j], p[j]); } });
  const c = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2];
  let sp = [(hi[0] - lo[0]) * (1 + 2 * pad), (hi[1] - lo[1]) * (1 + 2 * pad)];
  if (sp[0] / sp[1] < w / h) sp[0] = sp[1] * w / h; else sp[1] = sp[0] * h / w;
  return [[c[0] - sp[0] / 2, c[1] - sp[1] / 2], [c[0] + sp[0] / 2, c[1] + sp[1] / 2]];
}
// clip a-b to the rectangle (x0, y0, w, h); returns null when it misses entirely
function clipSeg(a, b, x0, y0, w, h){
  const d = [b[0] - a[0], b[1] - a[1]];
  let t0 = 0, t1 = 1;
  const edges = [[-d[0], a[0] - x0], [d[0], x0 + w - a[0]], [-d[1], a[1] - y0], [d[1], y0 + h - a[1]]];
  for (const [q, r] of edges){
    if (q === 0){ if (r < 0) return null; continue; }
    const t = r / q;
    if (q < 0) t0 = Math.max(t0, t); else t1 = Math.min(t1, t);
  }
  if (t0 > t1) return null;
  return [[a[0] + t0 * d[0], a[1] + t0 * d[1]], [a[0] + t1 * d[0], a[1] + t1 * d[1]]];
}
function lineThrough(sh, plot, base, dir, st, len){
  len = len || 60;
  const a = plot.px([base[0] - len * dir[0], base[1] - len * dir[1]]);
  const b = plot.px([base[0] + len * dir[0], base[1] + len * dir[1]]);
  const seg = clipSeg(a, b, plot.x0, plot.y0, plot.w, plot.h);
  if (seg) sh.push(S.e(seg[0][0], seg[0][1], seg[1][0], seg[1][1], {s:st, arrow:false, w:.065}));
  return sh;
}
function barRow(sh, x0, base, bw, gap, vals, sts, labs, maxh, top){
  const hi = top || Math.max.apply(null, vals.map(Math.abs));
  vals.forEach((v, i) => {
    const hgt = Math.max(.05, maxh * Math.abs(v) / hi), x = x0 + i * (bw + gap);
    sh.push(S.r(x, base - hgt, bw, hgt, sts[i] || 'ok', null));
    if (labs && labs[i] != null) sh.push(fitT(x + bw / 2, base - hgt - .20, labs[i], {c:COL.pale, fs:.24}));
  });
  return sh;
}
const dot2 = (a, b) => a[0] * b[0] + a[1] * b[1];

/* ================================================================ tab 1: centring */
const CODE_CENTRE = [
  'Xc = X - X.mean(axis=0)            # this line is the whole tab',
  'U, s, Vt = np.linalg.svd(Xc, full_matrices=False)',
  'components = Vt                    # rows are the directions',
  'explained_variance = s ** 2 / (n - 1)',
  '',
  '# without the first line, "variance" is measured about the',
  '# origin, so PC1 points at where the data IS.'
];
function centreStage(o){
  const sh = [], plot = new Plot(.45, .55, 4.9, 4.5,
                                 equalLim(PTS.concat([[0, 0]]), 4.9, 4.5));
  plot.frame(sh);
  dots(sh, plot, PTS, PTS.map(() => 'ok'), null, .10);
  const org = plot.px([0, 0]), mu = plot.px(PC_MEAN);
  sh.push(S.c(org[0], org[1], .10, 'soft', null));
  sh.push(fitT(org[0] + .45, org[1], {zh:'原點', en:'origin'}, {anchor:'start', c:COL.grey, fs:.24}));
  if (o.showC) lineThrough(sh, plot, PC_MEAN, PC_CENTRED, 'ok');
  if (o.showU) lineThrough(sh, plot, [0, 0], PC_UNCENTRED, 'bad');
  if (o.showMean){
    sh.push(S.e(mu[0] - .16, mu[1] - .16, mu[0] + .16, mu[1] + .16, {s:'hot', arrow:false, w:.055}));
    sh.push(S.e(mu[0] - .16, mu[1] + .16, mu[0] + .16, mu[1] - .16, {s:'hot', arrow:false, w:.055}));
    sh.push(fitT(mu[0] + .30, mu[1] + .62, {zh:'平均', en:'the mean'}, {anchor:'start', c:COL.orange, fs:.26}));
  }
  side(sh, 5.65, 1.15, o.side, .46);
  return caps(sh, o);
}
function centreFrames(){
  const F = new Frames();
  const steps = [
    {showMean:true, line:0,
     cap:{zh:'43 個點，離原點比它們自己的寬度還遠', en:'43 points, further from the origin than they are wide'},
     cap2:{zh:'平均在 (' + PC_MEAN[0] + ', ' + PC_MEAN[1] + ')', en:'the mean sits at (' + PC_MEAN[0] + ', ' + PC_MEAN[1] + ')'},
     side:[[{zh:'PCA 要找的是「最寬的方向」', en:'PCA looks for the widest direction'}, COL.purpleL, .28],
           [{zh:'問題是：相對於什麼量寬度？', en:'the question is: wide about what?'}, COL.grey, .26]],
     msg:{zh:'先看資料本身。這團點離原點相當遠，而它自己的擴散比那個距離小得多——這是真實資料的常態，因為欄位通常有個基準值。',
          en:'Start with the data. The cloud sits well away from the origin and its own spread is much smaller than that offset - which is what real columns usually look like, because they have a baseline.'}},
    {showMean:true, showC:true, line:1,
     cap:{zh:'減掉平均之後的 PC1：變異是相對於資料中心量的', en:'PC1 after subtracting the mean: variance measured about the data'},
     cap2:{zh:'這條線沿著點雲最長的那個方向', en:'the line runs along the longest direction of the cloud'},
     side:[[{zh:'第一主成分解釋 ' + CENTRE_NUM.evrC.toFixed(4), en:'the first component explains ' + CENTRE_NUM.evrC.toFixed(4)}, COL.tealL, .28],
           [{zh:'正確答案', en:'the right answer'}, COL.grey, .26]],
     msg:{zh:'這就是對的那條。它穿過平均，而且方向是點雲最長的軸——PCA 的定義就是這個。',
          en:'That is the correct one. It passes through the mean and runs along the longest axis of the cloud, which is what PCA is defined to find.'}},
    {showMean:true, showC:true, showU:true, line:5,
     cap:{zh:'忘記減平均之後的 PC1：變異是相對於原點量的', en:'PC1 without subtracting the mean: variance measured about the origin'},
     cap2:{zh:'它穿過原點，而且幾乎直接指向平均（差 ' + CENTRE_NUM.angMean.toFixed(1) + ' 度）',
           en:'it passes through the origin and points almost straight at the mean, ' + CENTRE_NUM.angMean.toFixed(1) + ' degrees off'},
     side:[[{zh:'跟正確答案差 ' + CENTRE_NUM.angPC.toFixed(1) + ' 度', en:CENTRE_NUM.angPC.toFixed(1) + ' degrees away from the real PC1'}, COL.red, .28],
           [{zh:'而它報出 ' + CENTRE_NUM.evrU.toFixed(4), en:'and it reports ' + CENTRE_NUM.evrU.toFixed(4)}, COL.red, .28],
           [{zh:'比正確的 ' + CENTRE_NUM.evrC.toFixed(4) + ' 還高', en:'which is higher than the correct ' + CENTRE_NUM.evrC.toFixed(4)}, COL.grey, .26]],
     msg:{zh:'紅線指的是「資料在哪裡」，不是「資料長什麼形狀」。真正危險的是最後那個數字：壞掉的版本報出 ' + CENTRE_NUM.evrU.toFixed(4) + '，比正確的 ' + CENTRE_NUM.evrC.toFixed(4) + ' 還漂亮——分母跟著分子一起被灌水了。大家真的會拿去報告的那個數字，犯了這個錯之後反而往上跑；不是每個數字都被騙（重建誤差就抓得到），它只是不會被放進投影片裡的那個。',
          en:'The red line points at where the data is, not at what shape it has. The dangerous part is the last number: the broken version reports ' + CENTRE_NUM.evrU.toFixed(4) + ' against the correct ' + CENTRE_NUM.evrC.toFixed(4) + '. Its denominator was inflated along with its numerator, so the number anyone actually reports goes UP when you make this mistake. Not every number is fooled - reconstruction error does catch it - it is simply not the one on the slide.'}}
  ];
  steps.forEach(s => F.push({
    shapes:centreStage(s),
    panels:[{lbl:'explained variance ratio', chips:[
      chip('centred ' + CENTRE_NUM.evrC.toFixed(4), s.showC ? 'ok' : 'dim'),
      chip('uncentred ' + CENTRE_NUM.evrU.toFixed(4), s.showU ? 'bad' : 'dim')]}],
    line:s.line, msg:s.msg}));
  return F.list;
}

/* ================================================================ tab 2: which direction matters */
const CODE_VAR = [
  'p = PCA().fit(X)',
  'p.explained_variance_ratio_   # [0.9833, 0.0167]',
  'Z = p.transform(X, k=1)       # keep 98% of the variance',
  '',
  '# PCA never sees y.  It is told to find the widest',
  '# direction and it does exactly that.'
];
function varStage(o){
  const sh = [], plot = new Plot(.45, .55, 4.4, 4.5, equalLim(HID, 4.4, 4.5));
  plot.frame(sh);
  dots(sh, plot, HID, HID_Y.map(v => (v === 1 ? 'act' : 'ok')), null, .10);
  if (o.pc != null){
    lineThrough(sh, plot, HID_MEAN, HID_PC[o.pc], o.pc === 0 ? 'bad' : 'hot');
  }
  // the 1-D projection as a strip
  if (o.pc != null){
    const d = HID_PC[o.pc];
    const z = HID.map(p => dot2([p[0] - HID_MEAN[0], p[1] - HID_MEAN[1]], d));
    const lo = Math.min.apply(null, z), hi = Math.max.apply(null, z);
    const bx = 5.35, bw = 3.9, by = 2.75;
    sh.push(S.e(bx, by, bx + bw, by, {s:'soft', arrow:false, w:.03}));
    z.forEach((v, i) => {
      const x = bx + bw * (v - lo) / (hi - lo);
      sh.push(S.c(x, by + (HID_Y[i] === 1 ? -.22 : .22), .07,
                  HID_Y[i] === 1 ? 'act' : 'ok', null));
    });
    sh.push(fitT(bx, by + .80, o.pc === 0 ? {zh:'投影到 PC1', en:'projected onto PC1'}
                                          : {zh:'投影到 PC2', en:'projected onto PC2'},
                 {anchor:'start', c:o.pc === 0 ? COL.red : COL.orange, fs:.30}));
    sh.push(fitT(bx, by - .80, {zh:'最好的一刀：' + HID_NUM.acc[o.pc].toFixed(4),
                                en:'best single cut: ' + HID_NUM.acc[o.pc].toFixed(4)},
                 {anchor:'start', c:o.pc === 0 ? COL.red : COL.orange, fs:.30}));
  }
  side(sh, 5.35, 1.20, o.side, .44);
  return caps(sh, o);
}
function varFrames(){
  const F = new Frames();
  const steps = [
    {pc:null, line:0,
     cap:{zh:'400 點兩類、旋轉過（圖上畫 90 個，數字算全部 400 個）',
          en:'400 points, two classes, rotated off the axes (90 drawn, numbers from all 400)'},
     cap2:{zh:'兩個類別沿著某一個方向分開', en:'the classes are separated along one direction'},
     side:[[{zh:'PCA 看不到顏色', en:'PCA never sees the colours'}, COL.purpleL, .28]],
     msg:{zh:'顏色是標籤。PCA 是非監督的，它看不到顏色——它只會看座標。接下來的問題是：它挑出來的方向，跟分開兩個顏色的那個方向，是不是同一個？',
          en:'The colours are the labels. PCA is unsupervised and never sees them; it only sees coordinates. The question is whether the direction it picks is the direction that separates the colours.'}},
    {pc:0, line:1,
     cap:{zh:'PC1：解釋了 ' + (100 * HID_NUM.evr[0]).toFixed(2) + '% 的變異',
          en:'PC1: explains ' + (100 * HID_NUM.evr[0]).toFixed(2) + '% of the variance'},
     cap2:{zh:'投影之後兩個顏色完全疊在一起', en:'after projecting, the two colours sit on top of each other'},
     side:[[{zh:'變異 ' + (100 * HID_NUM.evr[0]).toFixed(2) + '%', en:'variance ' + (100 * HID_NUM.evr[0]).toFixed(2) + '%'}, COL.red, .28],
           [{zh:'最好的一刀 ' + HID_NUM.acc[0].toFixed(4), en:'best cut ' + HID_NUM.acc[0].toFixed(4)}, COL.red, .28],
           [{zh:'跟擲銅板一樣', en:'a coin flip'}, COL.grey, .26]],
     msg:{zh:'PC1 抓到的是最寬的方向，而最寬的方向在這裡完全不帶資訊。把 98% 的變異留下來，就是把答案丟掉。',
          en:'PC1 is the widest direction, and here the widest direction carries no information at all. Keeping 98% of the variance means throwing the answer away.'}},
    {pc:1, line:2,
     cap:{zh:'PC2：只解釋了 ' + (100 * HID_NUM.evr[1]).toFixed(2) + '% 的變異',
          en:'PC2: explains only ' + (100 * HID_NUM.evr[1]).toFixed(2) + '% of the variance'},
     cap2:{zh:'而它把兩個顏色切得乾乾淨淨', en:'and it separates the two colours cleanly'},
     side:[[{zh:'變異 ' + (100 * HID_NUM.evr[1]).toFixed(2) + '%', en:'variance ' + (100 * HID_NUM.evr[1]).toFixed(2) + '%'}, COL.orange, .28],
           [{zh:'最好的一刀 ' + HID_NUM.acc[1].toFixed(4), en:'best cut ' + HID_NUM.acc[1].toFixed(4)}, COL.orange, .28],
           [{zh:'兩者的變異比是 ' + HID_NUM.ratio.toFixed(1) + ' 比 1', en:'the variance ratio is ' + HID_NUM.ratio.toFixed(1) + ' to 1'}, COL.grey, .26]],
     msg:{zh:'PCA 其實找對了「那一對方向」——旋轉是對的，訊號乾淨地落在它其中一根軸上。它只是用錯的鍵去排序。「保留 95% 的變異」在這裡剛好會把唯一有用的那一根丟掉，而 PCA 沒辦法知道，因為它從來看不到標籤。',
          en:'PCA did find the right pair of directions - the rotation is correct and the signal lies cleanly along one of its axes. It just sorted them by the wrong key. "Keep 95% of the variance" discards the only useful one here, and PCA cannot know any better, because it never sees the labels.'}}
  ];
  steps.forEach(s => F.push({
    shapes:varStage(s),
    panels:[{lbl:'PC1', chips:[chip((100 * HID_NUM.evr[0]).toFixed(2) + '% var', 'dim'),
                               chip('cut ' + HID_NUM.acc[0].toFixed(4), s.pc === 0 ? 'bad' : 'dim')]},
            {lbl:'PC2', chips:[chip((100 * HID_NUM.evr[1]).toFixed(2) + '% var', 'dim'),
                               chip('cut ' + HID_NUM.acc[1].toFixed(4), s.pc === 1 ? 'ok' : 'dim')]}],
    line:s.line, msg:s.msg}));
  return F.list;
}

/* ================================================================ tab 3: conditioning */
const CODE_COND = [
  '# the same mathematics, two algorithms',
  'U, s, Vt = np.linalg.svd(X - X.mean(0))      # singular values',
  '',
  'C = Xc.T @ Xc / (n - 1)                      # covariance',
  'w, V = np.linalg.eigh(C)                     # eigenvalues',
  '',
  '# cond(C) = cond(X) ** 2, and float64 has about 16 digits',
  '# to spend.  Past cond(X) of 1e8 there is nothing left.'
];
function condFrames(){
  const F = new Frames();
  const X0 = 1.15, WD = 7.2, Y0 = 1.25, HT = 2.9;
  const LO = -11, HI = -2;
  const yv = v => Y0 + HT * (v - LO) / (HI - LO);
  COND.forEach((_, step) => {
    const sh = [];
    sh.push(S.e(X0, Y0, X0 + WD, Y0, {s:'soft', arrow:false, w:.03}));
    sh.push(S.e(X0, Y0, X0, Y0 + HT, {s:'soft', arrow:false, w:.03}));
    for (let g = -10; g <= -2; g += 2)
      sh.push(fitT(X0 - .22, yv(g), '1e' + g, {anchor:'end', c:COL.grey, fs:.24}));
    const xs = COND.map((_, i) => X0 + WD * i / (COND.length - 1));
    COND.forEach((r, i) => sh.push(fitT(xs[i], Y0 - .30, '1e' + Math.round(Math.log10(r.eps)),
                                        {c:COL.grey, fs:.22})));
    for (let i = 0; i <= step; i++){
      const r = COND[i];
      const ySvd = yv(Math.log10(r.svd));
      sh.push(S.c(xs[i], ySvd, .11, 'ok', null));
      if (i > 0){
        const p = COND[i - 1];
        sh.push(S.e(xs[i - 1], yv(Math.log10(p.svd)), xs[i], ySvd, {s:'ok', arrow:false, w:.06}));
      }
      const yEig = r.eig > 0 ? yv(Math.log10(r.eig)) : Y0 + .05;
      sh.push(S.c(xs[i], yEig, .09, r.eig > 0 ? 'act' : 'bad', null));
      if (i > 0){
        const p = COND[i - 1];
        sh.push(S.e(xs[i - 1], p.eig > 0 ? yv(Math.log10(p.eig)) : Y0 + .05, xs[i], yEig,
                    {s:r.eig > 0 ? 'act' : 'bad', arrow:false, w:.045}));
      }
      if (r.eig === 0) sh.push(S.r(xs[i] - .22, Y0 - .16, .44, .42, 'bad', null));
    }
    const cur = COND[step];
    side(sh, X0 + .25, Y0 + HT - .30, [
      [{zh:'teal：從 SVD 得到的最小奇異值', en:'teal: the smallest singular value, from the SVD'}, COL.tealL, .26],
      [{zh:'purple：從共變異數矩陣的 eigh 得到的', en:'purple: the same number, via eigh of the covariance'}, COL.purple, .26]
    ], .40);
    caps(sh, {
      cap:{zh:'eps = 1e' + Math.round(Math.log10(cur.eps)) + '：cond(X) ' + cur.cond.toExponential(1) +
              '，cond(cov) ' + cur.ccond.toExponential(1),
           en:'eps = 1e' + Math.round(Math.log10(cur.eps)) + ': cond(X) ' + cur.cond.toExponential(1) +
              ', cond(cov) ' + cur.ccond.toExponential(1)},
      cap2:cur.eig > 0
        ? {zh:'SVD 給 ' + cur.svd.toExponential(3) + '，eigh 給 ' + cur.eig.toExponential(3),
           en:'the SVD gives ' + cur.svd.toExponential(3) + ', eigh gives ' + cur.eig.toExponential(3)}
        : {zh:'SVD 給 ' + cur.svd.toExponential(3) + '，eigh 給剛好 0',
           en:'the SVD gives ' + cur.svd.toExponential(3) + ', eigh gives exactly 0'}
    });
    F.push({shapes:sh,
      panels:[{lbl:'from the SVD', chips:[chip(cur.svd.toExponential(3), 'ok')]},
              {lbl:'from eigh', chips:[chip(cur.eig > 0 ? cur.eig.toExponential(3) : 'exactly 0',
                                            cur.eig > 0 ? 'act' : 'bad')]}],
      line:cur.eig > 0 ? 1 : 6,
      msg:step === 0
        ? {zh:'兩條路在數學上完全一樣：共變異數矩陣的特徵值就是奇異值的平方除以 n−1。在條件數還小的時候，兩個算法也給一樣的答案。',
           en:'The two routes are the same mathematics: the eigenvalues of the covariance are the singular values squared over n - 1. While the condition number is small, the two algorithms also give the same answer.'}
        : (cur.eig === 0
           ? {zh:'共變異數那條路回傳剛好 0，而真正的值是 ' + cur.svd.toExponential(3) + '。一個真的有變異的方向被四捨五入掉了。理由一行就講完：float64 大約 16 位數，而 cond(cov) = cond(X)²，所以 cond(X) 一過 1e8，平方之後的小奇異值就掉到雜訊底下。',
              en:'The covariance route returns exactly 0 where the true value is ' + cur.svd.toExponential(3) + '. A direction with real variance in it has been rounded out of existence. The reason is one line: float64 carries about 16 digits and cond(cov) = cond(X) squared, so once cond(X) passes 1e8 the small singular values sit below the noise floor of the squared problem.'}
           : {zh:'條件數再往上一階。注意 purple 已經開始落後——它丟掉的位數正好是 teal 的兩倍。',
              en:'One more order of magnitude. Notice that purple has already started to drift: it is losing digits at twice the rate.'})});
  });
  return F.list;
}

/* ================================================================ tab 4: truncated SVD */
const CODE_RANK = [
  'U, s, Vt = np.linalg.svd(M, full_matrices=False)',
  'Mk = (U[:, :k] * s[:k]) @ Vt[:k]',
  '',
  '# Eckart-Young: this is the best rank-k approximation',
  '# in Frobenius norm, and the error is exactly',
  '#   sqrt(sum of the discarded singular values squared)'
];
const RANK_KS = ['1', '2', '3', '5', '12'];
function rankFrames(){
  const F = new Frames();
  let lo = Infinity, hi = -Infinity;
  MAT.forEach(r => r.forEach(v => { lo = Math.min(lo, v); hi = Math.max(hi, v); }));
  const grid = (sh, M, x0, y0, cw) => {
    M.forEach((row, i) => row.forEach((v, j) => {
      const t = (v - lo) / (hi - lo);
      sh.push(S.r(x0 + j * cw, y0 + i * cw, cw * .92, cw * .92,
                  t > .72 ? 'ok' : (t > .48 ? 'act' : (t > .26 ? 'idle' : 'done')), null));
    }));
    return sh;
  };
  RANK_KS.forEach((kk, step) => {
    const rec = MAT_REC[kk], sh = [];
    const CW = .26;
    grid(sh, MAT, .60, 1.35, CW);
    grid(sh, rec.m, 4.05, 1.35, CW);
    sh.push(fitT(.60 + 6 * CW, 4.96, {zh:'原始矩陣', en:'the original matrix'}, {c:COL.grey, fs:.28}));
    sh.push(fitT(4.05 + 6 * CW, 4.96, {zh:'前 ' + kk + ' 個奇異值重建', en:'rebuilt from ' + kk + ' singular values'},
                 {c:COL.orange, fs:.28}));
    side(sh, 7.55, 1.70, [
      [{zh:'k = ' + kk, en:'k = ' + kk}, COL.purpleL, .32],
      [{zh:'誤差 ' + rec.err.toFixed(3), en:'error ' + rec.err.toFixed(3)}, COL.tealL, .28],
      [{zh:'預測 ' + rec.pred.toFixed(3), en:'predicted ' + rec.pred.toFixed(3)}, COL.orange, .28],
      [{zh:(100 * rec.kept).toFixed(1) + '% 的範數平方', en:(100 * rec.kept).toFixed(1) + ' % of |A|_F^2'}, COL.grey, .26]
    ], .46);
    caps(sh, {
      cap:{zh:'k = ' + kk + '：Frobenius 誤差 ' + rec.err.toFixed(4) + '，丟掉的尾巴開根號 ' + rec.pred.toFixed(4),
           en:'k = ' + kk + ': Frobenius error ' + rec.err.toFixed(4) + ', sqrt of the discarded tail ' + rec.pred.toFixed(4)},
      cap2:{zh:'兩個數字一樣，這就是 Eckart-Young', en:'the two numbers are the same - that is Eckart-Young'}});
    F.push({shapes:sh,
      panels:[{lbl:'rank k', chips:[chip(kk, 'hot')]},
              {lbl:'error / predicted', chips:[chip(rec.err.toFixed(4), 'ok'),
                                               chip(rec.pred.toFixed(4), 'act')]},
              {lbl:'|A|_F^2 kept', chips:[chip((100 * rec.kept).toFixed(1) + '%', 'dim')]}],
      line:1,
      msg:step === RANK_KS.length - 1
        ? {zh:'k 等於矩陣的秩時誤差歸零，因為什麼都沒丟。中間這條曲線是「保住了多少 Frobenius 範數平方」，而在置中過的資料上，那就正好是 explained variance ratio——同一個比例，只是停在統計之前一步。這也正是這個詞容易被誤用的原因：它講的是「把矩陣重建得多好」，跟「把你要的東西重建得多好」完全是兩件事。',
           en:'At k equal to the rank the error is zero, because nothing was discarded. The curve in between is the share of the squared Frobenius norm kept, and on centred data that is exactly the explained-variance ratio - the same ratio, one step before the statistics. It is also why the phrase gets misused: it is a statement about reconstructing the matrix, not about reconstructing the thing you wanted.'}
        : {zh:'k = ' + kk + ' 保住了 ' + (100 * rec.kept).toFixed(1) + '% 的 Frobenius 範數平方（這個矩陣沒置中，所以那不等於「變異」；置中過才是 explained variance ratio），誤差 ' + rec.err.toFixed(4) + '，而且剛好等於被丟掉的奇異值平方和開根號。這個等式對每一個 k 都成立，不是巧合。',
           en:'k = ' + kk + ' keeps ' + (100 * rec.kept).toFixed(1) + '% of the squared Frobenius norm - this matrix is not centred, so that is not the same as variance; on centred data it would be the explained-variance ratio - at an error of ' + rec.err.toFixed(4) + ', which is exactly the square root of the sum of the discarded singular values squared. That identity holds at every k; it is not a coincidence.'}});
  });
  return F.list;
}

/* ================================================================ tab 5: t-SNE running */
const CODE_TSNE = [
  'P = joint_P(X, perplexity)        # gaussian, one sigma per row',
  'Y = rng.standard_normal((n, 2)) * 1e-4',
  'for _ in range(n_iter):',
  '    num = 1 / (1 + pairwise_sq_dists(Y))    # Student-t',
  '    np.fill_diagonal(num, 0.0)    # not its own neighbour',
  '    Q = num / num.sum()',
  '    W = (P - Q) * num',
  '    grad = 4 * (Y * W.sum(1)[:, None] - W @ Y)',
  '    Y -= lr * grad                # the embedding IS the parameter'
];
function tsneFrames(){
  const F = new Frames();
  TSNE_SNAPS.forEach((Y, step) => {
    const sh = [], plot = new Plot(.50, .55, 5.4, 4.5, equalLim(Y, 5.4, 4.5, .12));
    plot.frame(sh);
    dots(sh, plot, Y, TSNE_Y.map(v => ['ok', 'act', 'hot'][v]), null, .11);
    side(sh, 6.05, 1.25, [
      [{zh:'第 ' + TSNE_ITERS[step] + ' 次迭代', en:'iteration ' + TSNE_ITERS[step]}, COL.purpleL, .30],
      [{zh:'KL ' + TSNE_KL[step].toFixed(4), en:'KL ' + TSNE_KL[step].toFixed(4)}, COL.tealL, .28],
      [{zh:'60 點、3 群', en:'60 points, 3 blobs'}, COL.grey, .26],
      [{zh:'perplexity 15', en:'perplexity 15'}, COL.grey, .26],
      [{zh:'顏色是真實分群', en:'the colours are the truth'}, COL.grey, .26]
    ], .42);
    caps(sh, {
      cap:step === 0
        ? {zh:'起點：所有的點都擠在原點附近的一小團雜訊裡', en:'the start: every point sits in a tiny blob of noise at the origin'}
        : {zh:'第 ' + TSNE_ITERS[step] + ' 次迭代，KL ' + TSNE_KL[step].toFixed(4),
           en:'iteration ' + TSNE_ITERS[step] + ', KL ' + TSNE_KL[step].toFixed(4)},
      cap2:step < 4
        ? {zh:'前 100 次迭代把 P 乘上 12，先把群推開',
           en:'the first 100 iterations multiply P by 12 to push the groups apart'}
        : {zh:'之後用真正的 P，把每一群自己整理好',
           en:'after that the real P takes over and each group settles'}});
    F.push({shapes:sh,
      panels:[{lbl:'iteration', chips:[chip(String(TSNE_ITERS[step]), 'hot')]},
              {lbl:'KL(P || Q)', chips:[chip(TSNE_KL[step].toFixed(4), 'ok')]}],
      line:step === 0 ? 1 : 6,
      msg:step === 0
        ? {zh:'t-SNE 沒有在配一個函數：被優化的東西就是這 60 個點在平面上的座標本身。所以它沒有 transform()，新的點只能整份重跑。',
           en:'t-SNE is not fitting a function: the thing being optimised is the 60 pairs of coordinates themselves. That is why there is no transform(), and why a new point means re-running everything.'}
        : (step === TSNE_SNAPS.length - 1
           ? {zh:'KL 從 ' + TSNE_KL[0].toFixed(4) + ' 降到 ' + TSNE_KL[step].toFixed(4) + '，而原本最近的 10 個鄰居有 ' + (100 * TSNE_RECALL).toFixed(0) + '% 還在圖上是鄰居。那個數字是這張圖唯一有資格宣稱的東西——下一個分頁是它沒有資格宣稱的那些。',
              en:'KL fell from ' + TSNE_KL[0].toFixed(4) + ' to ' + TSNE_KL[step].toFixed(4) + ', and ' + (100 * TSNE_RECALL).toFixed(0) + '% of each point’s ten nearest neighbours are still its neighbours on the page. That number is the only thing this picture is entitled to claim - the next tab is about everything it is not.'}
           : {zh:'梯度把 P 大的點對拉近、Q 大的點對推開。低維用的是 Student-t：尾巴很重，所以中等距離的一對可以被放得很開，不會全部擠在中間。',
              en:'The gradient pulls together the pairs where P is large and pushes apart the pairs where Q is large. The low-dimensional kernel is a Student-t: its tail is heavy, so a moderately distant pair is allowed to sit far apart instead of everything piling into the middle.'})});
  });
  return F.list;
}

/* ================================================================ tab 6: what the map does not say */
const CODE_READ = [
  'KL(P || Q) = sum_ij p_ij * log(p_ij / q_ij)',
  '',
  '# a sum over PAIRS of points.  There is no term named',
  '# "how wide is this cluster" and none named "how far apart',
  '# are these two clusters", so nothing in the optimisation was',
  '# ever trying to get either of them right.'
];
function readFrames(){
  const F = new Frames();
  const steps = [0, 1, 2];
  steps.forEach(step => {
    const sh = [];
    if (step === 0){
      const BW = .62, GAP = .34, BASE = 4.10, MAXH = 2.3;
      const top = Math.max.apply(null, READ.oSpread);
      barRow(sh, 1.00, BASE, BW, GAP, READ.oSpread, ['ok', 'hot', 'act'],
             READ.oSpread.map(v => v.toFixed(2)), MAXH, top);
      barRow(sh, 5.55, BASE, BW, GAP, READ.eSpread, ['ok', 'hot', 'act'],
             READ.eSpread.map(v => v.toFixed(2)), MAXH, top);
      [[1.00, {zh:'資料裡每一群的寬度', en:'width of each blob in the data'}],
       [5.55, {zh:'圖上每一群的寬度', en:'width of each blob on the page'}]].forEach(([x, t]) =>
        sh.push(fitT(x + 1.4, 4.62, t, {c:COL.pale, fs:.28})));
      for (let i = 0; i < 3; i++){
        sh.push(fitT(1.00 + i * (BW + GAP) + BW / 2, BASE + .32, 'blob ' + i, {c:COL.grey, fs:.22}));
        sh.push(fitT(5.55 + i * (BW + GAP) + BW / 2, BASE + .32, 'blob ' + i, {c:COL.grey, fs:.22}));
      }
      const ro = Math.max.apply(null, READ.oSpread) / Math.min.apply(null, READ.oSpread);
      const re = Math.max.apply(null, READ.eSpread) / Math.min.apply(null, READ.eSpread);
      side(sh, 1.00, 1.20, [
        [{zh:'資料裡最寬 / 最窄 = ' + ro.toFixed(2), en:'widest / narrowest in the data = ' + ro.toFixed(2)}, COL.tealL, .30],
        [{zh:'圖上最寬 / 最窄 = ' + re.toFixed(2), en:'widest / narrowest on the page = ' + re.toFixed(2)}, COL.red, .30]
      ], .48);
      caps(sh, {cap:{zh:'中間那一群在資料裡寬十倍，在圖上跟其他兩群一樣大',
                     en:'the middle blob is ten times wider in the data and the same size on the page'},
                cap2:{zh:'因為每一列都自己挑了一個 sigma 去湊同一個 perplexity',
                      en:'because every row chose its own sigma to reach the same perplexity'}});
      F.push({shapes:sh,
        panels:[{lbl:'spread in the data', chips:READ.oSpread.map(v => chip(v.toFixed(2), 'ok'))},
                {lbl:'spread on the page', chips:READ.eSpread.map(v => chip(v.toFixed(2), 'bad'))}],
        line:3,
        msg:{zh:'每一列自己挑寬度，正是 t-SNE 應付「不同密度的區域」的方法——也正是為什麼輸出沒辦法報告密度。圖上一群看起來有多大，跟它在資料裡有多大沒有關係。',
             en:'The per-row width is how t-SNE copes with regions of different density - and exactly why the output cannot report density. How big a group looks on the page has nothing to do with how big it is in the data.'}});
    } else if (step === 1){
      const BW = .66, GAP = .42, BASE = 4.20, MAXH = 2.5;
      const trueR = READ.dist.map(d => d[0]), pageR = READ.dist.map(d => d[1]);
      barRow(sh, 1.00, BASE, BW, GAP, trueR, trueR.map(() => 'ok'),
             trueR.map(v => v.toFixed(2)), MAXH, Math.max.apply(null, trueR));
      barRow(sh, 5.35, BASE, BW, GAP, pageR, pageR.map(() => 'bad'),
             pageR.map(v => v.toFixed(2)), MAXH, Math.max.apply(null, trueR));
      sh.push(fitT(2.40, 4.70, {zh:'資料裡的距離比', en:'distance ratio in the data'}, {c:COL.tealL, fs:.28}));
      sh.push(fitT(6.75, 4.70, {zh:'圖上的同一個比', en:'the same ratio on the page'}, {c:COL.red, fs:.28}));
      side(sh, 1.00, 1.30, [
        [{zh:'四個設定裡最遠的那一群，遠了 ' + trueR[3].toFixed(2) + ' 倍', en:'in the last setup one blob is ' + trueR[3].toFixed(2) + ' times further away'}, COL.tealL, .30],
        [{zh:'而圖上只有 ' + pageR[3].toFixed(2) + ' 倍', en:'and on the page it is ' + pageR[3].toFixed(2) + ' times'}, COL.red, .30]
      ], .48);
      caps(sh, {cap:{zh:'真實距離比從 2 拉到 27，圖上的比一路都是 1.0 左右',
                     en:'the true ratio goes from 2 to 27 and the ratio on the page stays near 1.0'},
                cap2:{zh:'群跟群之間的距離在這張圖上沒有意義',
                      en:'between-cluster distance on this page carries no information'}});
      F.push({shapes:sh,
        panels:[{lbl:'true ratio', chips:trueR.map(v => chip(v.toFixed(2), 'ok'))},
                {lbl:'on the page', chips:pageR.map(v => chip(v.toFixed(2), 'bad'))}],
        line:4,
        msg:{zh:'這裡沒有 bug。KL(P||Q) 是在一對一對的點上加總，而整個式子裡沒有任何一項叫做「這兩群之間隔多遠」，所以優化從頭到尾就沒有在管這件事。',
             en:'There is no bug here. KL(P || Q) is a sum over pairs of points, and no term in it asks how far apart two clusters are, so nothing in the optimisation was ever trying to get that right.'}});
    } else {
      const BW = .54, GAP = .26, BASE = 4.20, MAXH = 2.4;
      const vals = [READ.silRaw].concat(READ.silEmb);
      barRow(sh, 1.00, BASE, BW, GAP, vals, ['ok', 'bad', 'bad', 'bad', 'bad'],
             vals.map(v => v.toFixed(3)), MAXH, 0.45);
      ['the data', 'p=5', 'p=15', 'p=30', 'p=50'].forEach((t, i) =>
        sh.push(fitT(1.00 + i * (BW + GAP) + BW / 2, BASE + .32, t, {c:COL.grey, fs:.21})));
      sh.push(fitT(2.60, 4.70, {zh:'silhouette：這裡真的有分群嗎？', en:'silhouette: is there any grouping here?'},
                   {c:COL.pale, fs:.28}));
      barRow(sh, 5.95, BASE, 1.0, .70, [READ.corBlobs, READ.corNoise], ['ok', 'bad'],
             [READ.corBlobs.toFixed(3), READ.corNoise.toFixed(3)], MAXH, 1.0);
      ['blobs', 'noise'].forEach((t, i) =>
        sh.push(fitT(5.95 + i * 1.70 + .50, BASE + .32, t, {c:COL.grey, fs:.22})));
      sh.push(fitT(7.30, 4.70, {zh:'三次執行彼此同意嗎？', en:'do three runs agree with each other?'},
                   {c:COL.pale, fs:.28}));
      side(sh, 1.00, 1.30, [
        [{zh:'180 個點的 10 維純高斯雜訊', en:'180 points of 10-D gaussian noise'}, COL.grey, .26],
        [{zh:'資料本身 ' + READ.silRaw.toFixed(3) + '，四張圖分別是它的 4.23 到 4.56 倍',
          en:'the data scores ' + READ.silRaw.toFixed(3) + '; the four pictures of it score 4.23 to 4.56 times that'}, COL.red, .28]
      ], .46);
      caps(sh, {cap:{zh:'純雜訊的圖上有三群，而且換個種子就換一組',
                     en:'the picture of pure noise has three groups, and a different seed gives different ones'},
                cap2:{zh:'真的結構換種子還在（' + READ.corBlobs.toFixed(3) + '），假的不會（' + READ.corNoise.toFixed(3) + '）',
                      en:'real structure survives the seed (' + READ.corBlobs.toFixed(3) + '), invented structure does not (' + READ.corNoise.toFixed(3) + ')'}});
      F.push({shapes:sh,
        panels:[{lbl:'silhouette', chips:[chip('data ' + READ.silRaw.toFixed(3), 'ok')]
                   .concat(READ.silEmb.map(v => chip(v.toFixed(3), 'bad')))},
                {lbl:'agreement between runs', chips:[chip('blobs ' + READ.corBlobs.toFixed(3), 'ok'),
                                                      chip('noise ' + READ.corNoise.toFixed(3), 'bad')]}],
        line:5,
        msg:{zh:'silhouette 接近 0 的意思是「這個分群是隨便切的」。資料本身 ' + READ.silRaw.toFixed(3) + '，而四個 perplexity 下的圖分別是它的 4.23 到 4.56 倍——這是 Day 47「k-means 在均勻亂數上也會給三群」再往前一層：這個方法本來就是用來生出群的，所以它生出群。抓到它的測試只要多跑一次：真的結構換個隨機種子還在，假的不會。',
             en:'A silhouette near 0 means the grouping is arbitrary. The data scores ' + READ.silRaw.toFixed(3) + ' and the four pictures of it score 4.23 to 4.56 times that. This is Day 47\'s k-means on uniform noise one layer further along: the method is built to produce groups, so it produces groups. The test that catches it costs one more run - real structure survives a different random seed, invented structure does not.'}});
    }
  });
  return F.list;
}

/* ================================================================ tab 7: LeetCode */
const CODE_LC = [
  'nums.sort()',
  'def count(m):                    # pairs at most m apart',
  '    total = j = 0',
  '    for i in range(n):',
  '        while nums[i] - nums[j] > m: j += 1',
  '        total += i - j',
  '    return total',
  'lo, hi = 0, nums[-1] - nums[0]   # lo = 0, not 1',
  'while lo < hi:',
  '    mid = (lo + hi) // 2',
  '    if count(mid) >= k: hi = mid',
  '    else: lo = mid + 1'
];
function lcFrames(v){
  const F = new Frames();
  if (v === 0){
    const X0 = 1.00, CW = 1.20;
    LC719.steps.forEach((st, step) => {
      const sh = [];
      LC719.nums.forEach((val, i) => {
        sh.push(box(X0 + i * CW, 3.60, CW - .14, .78, 'idle', String(val), .34));
      });
      sh.push(fitT(X0, 4.74, {zh:'排序後的 nums', en:'nums, sorted'}, {anchor:'start', c:COL.grey, fs:.26}));
      // the candidate distance axis
      const AX = 1.00, AW = 7.4, span = LC719.nums[LC719.nums.length - 1] - LC719.nums[0];
      sh.push(S.e(AX, 2.30, AX + AW, 2.30, {s:'soft', arrow:false, w:.035}));
      for (let m = 0; m <= span; m++)
        sh.push(fitT(AX + AW * m / span, 1.94, String(m), {c:COL.grey, fs:.20}));
      const px = m => AX + AW * m / span;
      sh.push(S.r(px(st.lo) - .10, 2.16, Math.max(.20, px(st.hi) - px(st.lo) + .20), .28,
                  'act', null));
      if (st.mid != null){
        sh.push(S.c(px(st.mid), 2.30, .20, 'hot', null));
        sh.push(fitT(px(st.mid), 2.84, 'mid = ' + st.mid, {c:COL.orange, fs:.28}));
      }
      sh.push(fitT(px(st.lo), 1.52, 'lo = ' + st.lo, {c:COL.purpleL, fs:.26}));
      sh.push(fitT(px(st.hi), 1.14, 'hi = ' + st.hi, {c:COL.purpleL, fs:.26}));
      const done = st.mid == null;
      side(sh, 1.00, .50, [
        [done ? {zh:'答案 = ' + LC719.answer, en:'answer = ' + LC719.answer}
              : {zh:'count(' + st.mid + ') = ' + st.cnt + (st.go ? ' >= k，往左收' : ' < k，往右收'),
                 en:'count(' + st.mid + ') = ' + st.cnt + (st.go ? ' >= k, keep the left half' : ' < k, keep the right half')},
         done ? COL.tealL : COL.pale, .30]
      ], .40);
      caps(sh, {
        cap:done ? {zh:'收斂：第 ' + LC719.k + ' 小的成對距離是 ' + LC719.answer,
                    en:'converged: the ' + LC719.k + 'th smallest pair distance is ' + LC719.answer}
                 : {zh:'k = ' + LC719.k + '，目前的候選區間是 [' + st.lo + ', ' + st.hi + ']',
                    en:'k = ' + LC719.k + ', the surviving range of answers is [' + st.lo + ', ' + st.hi + ']'},
        cap2:{zh:'全部 15 個距離：' + LC719.all.join(', '),
              en:'all 15 distances: ' + LC719.all.join(', ')}});
      F.push({shapes:sh,
        panels:[{lbl:'lo / hi', chips:[chip('lo ' + st.lo, 'act'), chip('hi ' + st.hi, 'act')]},
                {lbl:'count(mid)', chips:[chip(st.mid == null ? '-' : String(st.cnt),
                                               st.mid == null ? 'dim' : 'hot')]}],
        line:done ? 11 : 9,
        msg:done
          ? {zh:'整個搜尋一次都沒有把那 15 個距離列出來。在 n = 10,000 時它們有 49,995,000 個，而二分搜尋只需要 log2(最大距離) 輪，每輪一次 O(n) 的滑動視窗。另外 lo 一定要從 0 開始：寫成 1 的話，只要答案是 0（也就是有重複值）就會錯，而那在手寫的幾個測資裡幾乎不會出現。',
             en:'The search never listed those 15 distances. At n = 10,000 there are 49,995,000 of them, while the binary search needs only log2(largest distance) rounds, each one a single O(n) sliding window. Note also that lo must start at 0: writing 1 breaks whenever the answer is 0 - that is, whenever a value repeats - which a handful of hand-written tests will never show you.'}
          : {zh:'count(m) 是「距離不超過 m 的點對有幾組」，它對 m 單調遞增，所以可以二分。注意它是用滑動視窗數的：因為陣列排好序，j 只會往右走，整個 count 是 O(n) 不是 O(n²)。',
             en:'count(m) is how many pairs are at most m apart. It is non-decreasing in m, which is what makes the answer binary-searchable. It is counted with a sliding window: because the array is sorted, j only ever moves right, so count is O(n) rather than O(n squared).'}});
    });
  } else {
    const G = [[1, 2, 34], [23, 5, 6], [4, 7, 8]];
    const rowsK = G.map(r => r.join(''));
    const colsK = [0, 1, 2].map(j => G.map(r => r[j]).join(''));
    const steps = [0, 1, 2];
    steps.forEach(step => {
      const sh = [], CW = 1.15, X0 = 1.30, Y0 = 1.90;
      G.forEach((row, i) => row.forEach((val, j) => {
        const inRow = step >= 1 && i === 0, inCol = step >= 1 && j === 0;
        sh.push(box(X0 + j * CW, Y0 + i * CW, CW - .14, CW - .20,
                    inRow && inCol ? 'hot' : (inRow ? 'ok' : (inCol ? 'act' : 'idle')),
                    String(val), .34));
      }));
      sh.push(fitT(X0 + 1.5 * CW, 1.42, {zh:'3 x 3 的格子', en:'a 3 x 3 grid'}, {c:COL.grey, fs:.26}));
      const lines = [];
      if (step >= 1){
        lines.push([{zh:'第 0 列 = (' + G[0].join(', ') + ')', en:'row 0 = (' + G[0].join(', ') + ')'}, COL.tealL, .30]);
        lines.push([{zh:'第 0 行 = (' + G.map(r => r[0]).join(', ') + ')', en:'column 0 = (' + G.map(r => r[0]).join(', ') + ')'}, COL.purple, .30]);
        lines.push([{zh:'是不同的序列', en:'these are different sequences'}, COL.grey, .26]);
      }
      if (step >= 2){
        lines.push([{zh:"串成字串都變成 '" + rowsK[0] + "'", en:"joined into a string both become '" + rowsK[0] + "'"}, COL.red, .30]);
        lines.push([{zh:'正確是 0，字串 key 給 1', en:'correct is 0; the string key gives 1'}, COL.red, .28]);
      }
      side(sh, 5.40, 2.10, lines, .50);
      caps(sh, {
        cap:step === 0 ? {zh:'LeetCode 2352：有幾組 (列, 行) 是一模一樣的序列？',
                          en:'LeetCode 2352: how many (row, column) pairs are the same sequence?'}
            : (step === 1 ? {zh:'第 0 列和第 0 行：不同的序列', en:'row 0 and column 0: different sequences'}
                          : {zh:"用 ''.join(map(str, row)) 當 key，兩個都變成 '" + rowsK[0] + "'",
                             en:"with ''.join(map(str, row)) as the key, both become '" + rowsK[0] + "'"}),
        cap2:{zh:'tuple 保留了元素的邊界，把數字黏成一串字串則把邊界丟掉',
              en:'a tuple keeps the boundaries; concatenated digits throw them away'}});
      F.push({shapes:sh,
        panels:[{lbl:'row 0 key', chips:[chip('(' + G[0].join(',') + ')', 'ok'),
                                         chip("'" + rowsK[0] + "'", step >= 2 ? 'bad' : 'dim')]},
                {lbl:'column 0 key', chips:[chip('(' + G.map(r => r[0]).join(',') + ')', 'act'),
                                            chip("'" + colsK[0] + "'", step >= 2 ? 'bad' : 'dim')]}],
        line:0,
        msg:step === 0
          ? {zh:'一列和一行相等的意思是「它們是同一串數字」，所以 key 必須是那串數字。tuple 可以，把數字直接黏成字串不行。',
             en:'A row and a column are equal when they are the same sequence of numbers, so the key has to be that sequence. A tuple keeps the element boundaries; gluing the digits into one string throws them away.'}
          : (step === 1
             ? {zh:'這兩個序列明顯不一樣：(1, 2, 34) 和 (1, 23, 4)。正確的答案是 0 組。',
                en:'The two sequences are plainly different: (1, 2, 34) and (1, 23, 4). The correct count is 0.'}
             : {zh:"黏成字串之後兩個都是 '" + rowsK[0] + "'，於是程式回報 1。這個 bug 有多大？在 1..99 的所有 9801 個二元組裡，只有 9072 個不同的字串 key，其中 729 個 key 被共用，每個剛好被兩個序列共用，所以 9801 個序列裡有 1458 個（14.9%）沒有唯一的 key。而兩個隨機抽的序列撞在一起的機率幾乎是 0——面積一成五，隨機測試卻幾乎踩不到，這就是會上線的 bug 的形狀。",
                en:"Joined, both become '" + rowsK[0] + "', so the code reports 1. How big is the bug? Among all 9801 two-element sequences with values in 1..99 there are only 9072 distinct joined keys. 729 keys are shared, each by exactly two sequences, so 1458 of the 9801 - 14.9% - are not uniquely keyed. Yet two randomly drawn sequences almost never collide. Fifteen percent of the input space and essentially no chance of stumbling into it, which is the exact shape of a bug that reaches production."})});
    });
  }
  return F.list;
}

/* ================================================================ meta */
const DAY_META = {
  title:{zh:'Day 49 — PCA、SVD 與 t-SNE', en:'Day 49 - PCA, SVD and t-SNE'},
  sub:{zh:'每一種降維都在回答一個你沒有問的問題，而出來的那張圖極容易被過度解讀。',
       en:'Every dimensionality reduction answers a question you did not ask, and the picture it returns is very easy to over-read.'},
  tabs:[
    {
      id:'centre', label:{zh:'忘了減平均', en:'forgetting the mean'},
      stage:{zh:'變異是相對於什麼量的？', en:'variance measured about what?'},
      view:VIEW,
      idea:{zh:'PCA 的第一行是 X - X.mean(axis=0)，而那一行就是整個分頁。沒有它的話，「變異」是相對於原點量的，於是第一主成分指向「資料在哪裡」而不是「資料長什麼形狀」——在這 43 個點上，沒減平均的 PC1 幾乎直接指著平均（差 ' + CENTRE_NUM.angMean.toFixed(1) + ' 度），跟正確的 PC1 差了 ' + CENTRE_NUM.angPC.toFixed(1) + ' 度。真正危險的是分數：壞掉的版本回報 ' + CENTRE_NUM.evrU.toFixed(4) + '，比正確的 ' + CENTRE_NUM.evrC.toFixed(4) + ' 還高，因為它現在量的那個偏移量比它應該量的擴散還大，分母跟著分子一起被灌水。大家真的會拿去報告的那個數字，犯了這個錯之後反而往上跑；不是每個數字都被騙（重建誤差就抓得到），它只是不會被放進投影片裡的那個。順帶一提，特徵值不重複時特徵向量只定義到正負號：v 和 -v 張出同一條線，兩個都對，所以同一份程式在不同版本或不同機器上可能給你一張翻過來的圖；程式裡要比較主成分就比 |u . v|，要給人看就把慣例釘死。',
            en:'The first line of PCA is X - X.mean(axis=0), and that line is this entire tab. Without it, variance is measured about the origin, so the first component points at where the data is rather than at what shape it has: on these 43 points the uncentred PC1 points almost straight at the mean (' + CENTRE_NUM.angMean.toFixed(1) + ' degrees off) and misses the real PC1 by ' + CENTRE_NUM.angPC.toFixed(1) + ' degrees. The dangerous part is the score: the broken version reports ' + CENTRE_NUM.evrU.toFixed(4) + ' against the correct ' + CENTRE_NUM.evrC.toFixed(4) + ', because the offset it is now measuring is bigger than the spread it should have been measuring and the denominator was inflated along with the numerator. The number anyone actually reports goes UP when you make this mistake; not every number is fooled, since reconstruction error does catch it, it is simply not the one on the slide. A related footnote: with a simple eigenvalue an eigenvector is defined only up to its sign, since v and -v span the same line and both are correct - which is why the same code on a different version or machine can hand you a flipped plot. Compare components with |u . v|, and pin a convention for anything a human will look at.'},
      legend:[['#3fe0dd', {zh:'資料點 / 正確的 PC1', en:'the points, and the correct PC1'}],
              ['#ff5c5c', {zh:'沒減平均的 PC1', en:'PC1 without centring'}],
              ['#ff9736', {zh:'資料的平均', en:'the mean of the data'}],
              ['#8fa3ac', {zh:'原點', en:'the origin'}]],
      get code(){ return CODE_CENTRE; },
      build:centreFrames
    },
    {
      id:'variance', label:{zh:'最寬 ≠ 最有用', en:'widest is not most useful'},
      stage:{zh:'PCA 被問的是哪個方向最寬，不是哪個方向有用',
             en:'PCA was asked which direction is widest, not which one is useful'},
      view:VIEW,
      idea:{zh:'400 個點、兩個類別，沿著某一個方向分開；另一個方向帶著大得多的變異，而且一點資訊都沒有；整個東西被旋轉過，所以沒有一個方向是座標軸。PC1 解釋了 ' + (100 * HID_NUM.evr[0]).toFixed(2) + '% 的變異，而用它做最好的一刀只能分對 ' + HID_NUM.acc[0].toFixed(4) + '——跟擲銅板一樣。PC2 只有 ' + (100 * HID_NUM.evr[1]).toFixed(2) + '%，卻分對 ' + HID_NUM.acc[1].toFixed(4) + '。兩者的變異比是 ' + HID_NUM.ratio.toFixed(1) + ' 比 1，而排序對「唯一有人會問的那個問題」來說剛好是反的。值得注意的是 PCA 其實找對了那一對方向：旋轉是對的，訊號乾乾淨淨地落在它其中一根軸上，它只是用錯的鍵去排序。「保留 95% 的變異」在這裡會把 PC2 丟掉，而 PCA 沒辦法做得更好——它是非監督的，從頭到尾沒看過標籤。',
            en:'400 points in two classes separated along one direction, with another direction carrying far more variance and no information, and the whole thing rotated so that neither one is an axis. PC1 explains ' + (100 * HID_NUM.evr[0]).toFixed(2) + '% of the variance and the best single cut along it separates the classes at ' + HID_NUM.acc[0].toFixed(4) + ' - a coin flip. PC2 holds ' + (100 * HID_NUM.evr[1]).toFixed(2) + '% and separates them at ' + HID_NUM.acc[1].toFixed(4) + '. The variance ratio between the two is ' + HID_NUM.ratio.toFixed(1) + ' to 1, and the ordering is exactly backwards for the only question anyone was going to ask. Note that PCA did find the right pair of directions - the rotation is correct and the signal lies cleanly along one of its axes - it only sorted them by the wrong key. "Keep 95% of the variance" discards PC2 here, and PCA could not have done better: it is unsupervised and never sees the labels.'},
      legend:[['#3fe0dd', {zh:'類別 0', en:'class 0'}],
              ['#9d6bff', {zh:'類別 1', en:'class 1'}],
              ['#ff5c5c', {zh:'PC1，最寬的方向', en:'PC1, the widest direction'}],
              ['#ff9736', {zh:'PC2，有答案的那個', en:'PC2, the one with the answer'}]],
      get code(){ return CODE_VAR; },
      build:varFrames
    },
    {
      id:'cond', label:{zh:'為什麼用 SVD', en:'why the SVD'},
      stage:{zh:'做出 X 轉置乘 X 會把條件數平方', en:'forming X transpose X squares the condition number'},
      view:VIEW,
      idea:{zh:'課本上的 PCA 是「建共變異數矩陣、做特徵分解」，而 numpy 和 scikit-learn 都走 SVD，理由是數值而不是數學。共變異數矩陣的特徵值就是奇異值的平方，所以它的條件數也是平方，而 float64 只有大約 16 位數可以花。這一列資料的奇異值正比於 1、eps、eps/2（基底另外乘了 √(n−1)），而把 eps 一路調小：到 eps = 1e-8，cond(X) 是 2e8、cond(cov) 是 4.2e16，共變異數那條路回傳的最小奇異值是**剛好 0**，而真正的值是 7.04e-08。一個真的有變異的方向被四捨五入掉了，沒有任何警告。同一個地方還有第二個症狀：eigh 會對一個半正定矩陣回傳負的特徵值（約 -1e-16），也就是負的變異數，開根號會得到 nan——所以從特徵值算標準差的程式必須夾在 0，而夾住正好把問題藏起來。',
            en:'The textbook PCA builds the covariance matrix and eigendecomposes it; numpy and scikit-learn both take the SVD instead, for numerical rather than mathematical reasons. The eigenvalues of the covariance are the singular values squared, so its condition number is squared too, and float64 only has about 16 digits to spend. This data has singular values proportional to 1, eps and eps/2 (the basis is scaled by sqrt(n - 1)), and as eps shrinks: at eps = 1e-8, cond(X) is 2e8, cond(cov) is 4.2e16, and the covariance route returns exactly 0 for a smallest singular value that is really 7.04e-08. A direction with real variance in it has been rounded out of existence, with no warning. There is a second symptom in the same place: eigh returns a small negative eigenvalue for a positive semidefinite matrix - a negative variance - and np.sqrt of it is nan, so code that takes standard deviations from eigenvalues has to clamp at zero, and clamping is what hides the problem.'},
      legend:[['#3fe0dd', {zh:'從 SVD 算出來的', en:'computed from the SVD'}],
              ['#9d6bff', {zh:'從共變異數的 eigh 算出來的', en:'computed via eigh of the covariance'}],
              ['#ff5c5c', {zh:'回傳剛好 0', en:'returned as exactly 0'}],
              ['#8fa3ac', {zh:'座標軸', en:'the axes'}]],
      get code(){ return CODE_COND; },
      build:condFrames
    },
    {
      id:'rank', label:{zh:'截斷的 SVD', en:'the truncated SVD'},
      stage:{zh:'Eckart-Young：截斷就是最好的低秩近似',
             en:'Eckart-Young: truncating is the best low-rank approximation'},
      view:VIEW,
      idea:{zh:'把 SVD 在第 k 個奇異值截斷，得到的就是 Frobenius 範數下最好的 rank-k 近似，而且誤差剛好等於「被丟掉的奇異值平方和」開根號。這不是近似的說法，是一個等式——在這個 12 x 12 的矩陣上，每一個 k 的實測誤差和預測值都對到小數點後四位。這個等式就是「解釋變異」這個詞的定義，也正是它那麼容易被誤用的原因：它講的是「把矩陣重建得多好」。「保留 95% 的變異」是在說「重建誤差的平方只剩 5%」，而不是在說「我關心的那個東西還留著 95%」——上一個分頁就是這兩件事分道揚鑣的樣子。順帶一提，真實系統很少做完整的 SVD：randomized SVD 只需要 k 個方向，成本大約是 O(n · d · k) 而不是 O(n · d²)。',
            en:'Truncate the SVD at the kth singular value and you have the best rank-k approximation in Frobenius norm, with an error exactly equal to the square root of the sum of the discarded singular values squared. That is not an approximation but an identity: on this 12 x 12 matrix the measured error and the prediction agree to four decimal places at every k. That identity is the definition of explained variance, and it is exactly why the phrase gets misused - it is a statement about reconstructing the matrix. "Keep 95% of the variance" says the squared reconstruction error is down to 5%; it does not say that 95% of the thing you care about survived, and the previous tab is what those two parting company looks like. As a footnote, real systems rarely compute the full SVD: randomized SVD only needs k directions and costs about O(n * d * k) rather than O(n * d squared).'},
      legend:[['#3fe0dd', {zh:'大的值', en:'large values'}],
              ['#9d6bff', {zh:'中間的值', en:'middling values'}],
              ['#ff9736', {zh:'重建出來的矩陣', en:'the rebuilt matrix'}],
              ['#2f5661', {zh:'小的值', en:'small values'}]],
      get code(){ return CODE_RANK; },
      build:rankFrames
    },
    {
      id:'tsne', label:{zh:'t-SNE 跑起來', en:'t-SNE, running'},
      stage:{zh:'被優化的就是這些點在平面上的座標本身',
             en:'the thing being optimised is the coordinates themselves'},
      view:VIEW,
      idea:{zh:'P：在原本的空間裡，用高斯把距離變成「鄰居機率」。每一列有自己的寬度，用二分搜尋調到那一列的 perplexity 落在你要的值上；perplexity 是 2 的熵次方，也就是「這個點實際上有幾個鄰居」。Q：在二維的圖上改用自由度 1 的 Student-t，它的尾巴很重，所以原空間裡中等距離的一對可以在紙上被放得很開——這就是「crowding problem」的解法，不然高維裡幾乎每一對都是中等距離，全部會擠在中間。然後用梯度下降最小化 KL(P || Q)。被優化的東西就是這 60 個點的座標本身，不是任何函數的參數，所以 t-SNE 沒有 transform()：新的點只能整份重跑，而那是每一輪 n 平方。前 100 次迭代把 P 乘以 12（early exaggeration），先把群推開再讓它們各自整理。最後 KL 從 ' + TSNE_KL[0].toFixed(4) + ' 降到 ' + TSNE_KL[TSNE_KL.length - 1].toFixed(4) + '，原本最近的 10 個鄰居有 ' + (100 * TSNE_RECALL).toFixed(0) + '% 還是鄰居——那個數字是這張圖唯一有資格宣稱的東西。',
            en:'P: in the original space, turn distances into neighbour probabilities with a gaussian. Every row gets its own width, found by binary search so that the perplexity of that row hits the number you asked for; perplexity is 2 to the power of the entropy, or "how many neighbours does this point effectively have". Q: on the 2-D page, use a Student-t with one degree of freedom instead. Its tail is heavy, so a pair that is moderately far apart in the original space may sit very far apart on the page - which is the fix for the crowding problem, since in high dimensions almost every pair is moderately far apart and they would otherwise all pile into the middle. Then minimise KL(P || Q) by gradient descent. What is being optimised is the 60 pairs of coordinates themselves, not the parameters of any function, which is why t-SNE has no transform(): a new point means re-running everything, and that is n squared per iteration. The first 100 iterations multiply P by 12 - early exaggeration - to push the groups apart before letting each settle. KL falls from ' + TSNE_KL[0].toFixed(4) + ' to ' + TSNE_KL[TSNE_KL.length - 1].toFixed(4) + ' and ' + (100 * TSNE_RECALL).toFixed(0) + '% of each point\'s ten nearest neighbours are still its neighbours, which is the only thing this picture is entitled to claim.'},
      legend:[['#3fe0dd', {zh:'第 0 群', en:'group 0'}],
              ['#9d6bff', {zh:'第 1 群', en:'group 1'}],
              ['#ff9736', {zh:'第 2 群', en:'group 2'}],
              ['#2f5661', {zh:'畫布', en:'the page'}]],
      get code(){ return CODE_TSNE; },
      build:tsneFrames
    },
    {
      id:'read', label:{zh:'這張圖沒說的事', en:'what the map does not say'},
      stage:{zh:'KL 裡沒有「群有多寬多遠」這一項', en:'KL has no term for cluster width or distance'},
      view:VIEW,
      idea:{zh:'三件事，都量得出來。一，群的大小沒有意義：中間那一群在資料裡寬十倍（最寬/最窄 = 10.56），在圖上只有 1.12 倍。原因是每一列自己挑 sigma 去湊同一個 perplexity，那正是 t-SNE 應付不同密度區域的方法，也正是為什麼輸出沒辦法報告密度。二，群跟群之間的距離沒有意義：把其中一群推到 2 倍、4 倍、8 倍、27 倍遠，圖上的比例一路都是 1.0 左右。三，連「有沒有群」都不一定有意義：180 個點的 10 維純高斯雜訊，資料本身的 silhouette 是 ' + READ.silRaw.toFixed(3) + '，而四個 perplexity 下的 t-SNE 圖分別是它的 4.23 到 4.56 倍。這三件事都不是 bug——KL(P || Q) 是在一對一對的點上加總，而整個式子裡沒有任何一項叫做「這群多寬」或「這兩群隔多遠」，所以優化從來沒有在試著把它們弄對。抓假結構的測試只要多跑一次：真的結構換個隨機種子還在（成對距離相關 ' + READ.corBlobs.toFixed(3) + '），假的不會（' + READ.corNoise.toFixed(3) + '）。',
            en:'Three things, all measurable. One, cluster size means nothing: the middle blob is ten times wider in the data (widest over narrowest = 10.56) and 1.12 times wider on the page. Every row picks its own sigma to reach the same perplexity, which is how t-SNE copes with regions of different density and exactly why the output cannot report density. Two, between-cluster distance means nothing: push one group to 2, 4, 8 and 27 times further away and the ratio on the page stays near 1.0 throughout. Three, even "are there groups at all" can mean nothing: on 180 points of 10-dimensional gaussian noise the data scores a silhouette of ' + READ.silRaw.toFixed(3) + ' while the four t-SNE pictures of it score 4.23 to 4.56 times that. None of this is a bug: KL(P || Q) is a sum over pairs of points, and it contains no term named how wide a cluster is or how far apart two clusters are, so the optimisation was never trying to get either right. The test that catches invented structure costs one more run - real structure survives a different seed (pairwise-distance correlation ' + READ.corBlobs.toFixed(3) + '), invented structure does not (' + READ.corNoise.toFixed(3) + ').'},
      legend:[['#3fe0dd', {zh:'資料裡量到的', en:'measured in the data'}],
              ['#ff5c5c', {zh:'圖上量到的', en:'measured on the page'}],
              ['#ff9736', {zh:'中間那一群', en:'the middle blob'}],
              ['#9d6bff', {zh:'第三群', en:'the third blob'}]],
      get code(){ return CODE_READ; },
      build:readFrames
    },
    {
      id:'lc', label:{zh:'LeetCode 719 與 2352', en:'LeetCode 719 and 2352'},
      stage:{zh:'n 平方對的成本，以及 A 轉置的 key', en:'the cost of n squared pairs, and the key for A transpose'},
      view:VIEW,
      variants:[{zh:'719 二分答案', en:'719, binary search the answer'},
                {zh:'2352 的 key', en:'2352, the key'}],
      idea:{zh:'今天兩半都建立在成對距離上，而那有 n(n-1)/2 個——n = 10,000 時是 49,995,000 個，也就是「你不被允許真的生出來」的那個數字。719 就是這個限制本身：把那些距離排序是不可能的，但 count(m) =「距離不超過 m 的點對有幾組」對 m 單調遞增，所以可以二分答案，而 count 本身因為陣列排好序，用滑動視窗是 O(n)。整個解法是 O(n log n + n log(最大距離))，一次都沒有把距離列出來。這裡的坑是 lo 要從 0 開始：寫成 1（「距離至少是 1」）只在答案是 0 的時候錯，而答案是 0 需要陣列裡有重複值——5,000 筆隨機測資裡有 316 筆（6.3%）不一樣，而幾個手寫的、值都相異的測資永遠看不到它。2352 則是關於 A 轉置，也就是今天另一半的材料（共變異數就是 X 轉置乘 X）。一列和一行相等的意思是「同一串數字」，所以 key 必須是那串數字：tuple 保留了元素的邊界，把數字黏成一串字串則把邊界丟掉——(1, 2, 34) 和 (1, 23, 4) 都會變成 "1234"。這個 bug 佔了輸入空間的 14.9%（1..99 的 9801 個二元組只對應 9072 個字串 key，729 個 key 被兩個序列共用，等於 1458 個序列沒有唯一的 key），而隨機抽兩個序列幾乎永遠撞不到。',
            en:'Both halves of today are built on pairwise distances, and there are n(n-1)/2 of them - 49,995,000 at n = 10,000, which is the number you are not allowed to materialise. 719 is that constraint by itself: sorting those distances is impossible, but count(m) - how many pairs are at most m apart - is non-decreasing in m, so the answer can be binary-searched, and count itself is O(n) with a sliding window because the array is sorted. The whole solution is O(n log n + n log(largest distance)) and never lists a single distance. The trap here is that lo must start at 0: writing 1, on the reasoning that a distance is at least 1, is wrong only when the answer is 0, which needs a repeated value - it differs on 316 of 5,000 random cases, 6.3%, and a handful of hand-written tests with distinct values will never show it. 2352 is about A transpose, which is the other half of today\'s material, since the covariance is X transpose times X. A row and a column are equal when they are the same sequence of numbers, so the key has to be that sequence: a tuple keeps the element boundaries, a string of concatenated digits throws them away - (1, 2, 34) and (1, 23, 4) both become "1234". That bug covers 14.9% of the input space (the 9801 two-element sequences with values in 1..99 map onto only 9072 distinct joined keys; 729 keys are shared by exactly two sequences each, so 1458 sequences are not uniquely keyed) and two randomly drawn sequences almost never collide.'},
      legend:[['#9d6bff', {zh:'還活著的候選區間 / 第 0 行', en:'the surviving range, and column 0'}],
              ['#ff9736', {zh:'目前的 mid / 兩者共用的格子', en:'the current mid, and the shared cell'}],
              ['#3fe0dd', {zh:'排序後的陣列 / 第 0 列', en:'the sorted array, and row 0'}],
              ['#8fa3ac', {zh:'候選距離的座標軸', en:'the axis of candidate distances'}]],
      get code(){ return CODE_LC; },
      build:lcFrames
    }
  ]
};

/* =========================================================================
   100 Days of Python - shared demo engine
   palette: teal/blue-green base, purple pointers, orange "current step"
   ========================================================================= */
const T = {
  title:{zh:DAY_META.title.zh, en:DAY_META.title.en},
  sub:{zh:DAY_META.sub.zh, en:DAY_META.sub.en},
  play:{zh:'▶ 播放', en:'▶ Play'}, pause:{zh:'❚❚ 暫停', en:'❚❚ Pause'},
  speed:{zh:'速度', en:'Speed'}, state:{zh:'演算法狀態', en:'Algorithm state'},
  code:{zh:'程式碼', en:'Code'}, idea:{zh:'重點', en:'The idea'}
};
let LANG = 'zh';
const tr = o => (o == null ? '' : (typeof o === 'string' ? o : (o[LANG] != null ? o[LANG] : o.en)));
const $ = id => document.getElementById(id);

function setLang(l){
  LANG = l;
  document.documentElement.lang = l === 'zh' ? 'zh-Hant' : 'en';
  $('btn-zh').classList.toggle('on', l === 'zh');
  $('btn-en').classList.toggle('on', l === 'en');
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const k = el.getAttribute('data-i18n');
    if (T[k]) el.textContent = tr(T[k]);
  });
  buildTabs(); render();
}

/* --------------------------------------------------------------- palette */
const STY = {
  idle :{fill:'#07293a', stroke:'#0a6b74', text:'#dff2f5', w:.045, glow:0},
  soft :{fill:'#052330', stroke:'#0a6b74', text:'#a8c8d0', w:.035, glow:0},
  hot  :{fill:'#3a2109', stroke:'#ff9736', text:'#ffbe6b', w:.075, glow:.65},
  act  :{fill:'#241542', stroke:'#9d6bff', text:'#c7a6ff', w:.070, glow:.55},
  ok   :{fill:'#08414a', stroke:'#3fe0dd', text:'#d9ffff', w:.070, glow:.5},
  done :{fill:'#062430', stroke:'#2f5661', text:'#7f9aa3', w:.035, glow:0},
  bad  :{fill:'#3a0d0d', stroke:'#ff5c5c', text:'#ff9a9a', w:.070, glow:.4},
  ghost:{fill:'none',    stroke:'#2f5661', text:'#7f9aa3', w:.035, glow:0, dash:'.12 .10'}
};
const COL = {teal:'#12b3b8', tealL:'#3fe0dd', purple:'#9d6bff', purpleL:'#c7a6ff',
             orange:'#ff9736', orangeL:'#ffbe6b', red:'#ff5c5c', grey:'#8fa3ac',
             pale:'#dff2f5', white:'#ffffff'};
const LEGEND = {
  hot:[COL.orange, {zh:'目前這一步', en:'current step'}],
  act:[COL.purple, {zh:'指標 / 走訪位置', en:'pointer / cursor'}],
  ok:[COL.tealL,  {zh:'完成 / 結果', en:'done / result'}],
  bad:[COL.red,   {zh:'失敗 / 要避開的寫法', en:'failure / the wrong way'}],
  done:['#2f5661', {zh:'已處理完', en:'already done'}],
  soft:['#0a6b74', {zh:'其他元素', en:'other items'}],
  idle:['#0a6b74', {zh:'尚未處理', en:'untouched'}],
  ghost:['#123f4d', {zh:'尚未處理', en:'not yet reached'}]
};
const leg = (...keys) => keys.map(k => LEGEND[k]);

/* ---------------------------------------------------------- frame buffer */
function Frames(){
  this.list = [];
  this.push = (o) => this.list.push({
    shapes:JSON.parse(JSON.stringify(o.shapes || [])),
    panels:JSON.parse(JSON.stringify(o.panels || [])),
    view:(o.view || null), line:(o.line == null ? 0 : o.line), msg:o.msg
  });
}
/* shape helpers - engines call these, the renderer just draws */
const S = {
  r:(x, y, w, h, s, lab, o) => Object.assign({t:'r', x:x, y:y, w:w, h:h, s:s || 'idle', lab:lab}, o || {}),
  c:(x, y, r, s, lab, o) => Object.assign({t:'c', x:x, y:y, r:r, s:s || 'idle', lab:lab}, o || {}),
  e:(x1, y1, x2, y2, o) => Object.assign({t:'e', x1:x1, y1:y1, x2:x2, y2:y2}, o || {}),
  t:(x, y, s, o) => Object.assign({t:'t', x:x, y:y, s:s}, o || {})
};
/* a labelled row of array cells; returns shapes */
function cellRow(vals, x0, y, w, h, opt){
  opt = opt || {};
  const out = [], st = opt.states || {};
  vals.forEach((v, i) => {
    out.push(S.r(x0 + i * w, y, w - (opt.gap == null ? .06 : opt.gap), h, st[i] || 'idle',
                 v == null ? '' : String(v), {fs:opt.fs || h * .52}));
    if (opt.index !== false)
      out.push(S.t(x0 + i * w + (w - .06) / 2, y + h + (opt.ilift || .34),
                   opt.labels ? opt.labels[i] : String(i),
                   {c:st[i] && st[i] !== 'idle' ? COL.orangeL : COL.grey, fs:opt.ifs || .30}));
  });
  if (opt.title) out.push(S.t(x0 - .22, y + h * .62, opt.title, {c:COL.tealL, fs:.32, anchor:'end'}));
  return out;
}
/* binary-tree layout from a heap-style array (index 0 = root, 2i+1 / 2i+2) */
function heapTreeShapes(arr, x0, y0, w, rowH, states, opt){
  opt = opt || {};
  const n = arr.length, out = [], R = opt.r || .34;
  const depth = i => Math.floor(Math.log2(i + 1));
  const maxD = n ? depth(n - 1) : 0;
  const px = i => {
    const d = depth(i), first = Math.pow(2, d) - 1, k = i - first;
    const slots = Math.pow(2, d), span = w;
    return x0 + span * (k + .5) / slots;
  };
  const py = i => y0 + depth(i) * rowH;
  for (let i = 1; i < n; i++){
    if (arr[i] == null) continue;
    const p = Math.floor((i - 1) / 2);
    out.push(S.e(px(p), py(p), px(i), py(i), {pad:R + .04,
      s:(states && (states[i] === 'hot' || states[i] === 'act')) ? states[i] : 'idle'}));
  }
  for (let i = 0; i < n; i++){
    if (arr[i] == null) continue;
    out.push(S.c(px(i), py(i), R, (states && states[i]) || 'idle', String(arr[i]), {fs:R * .95}));
    if (opt.showIndex)
      out.push(S.t(px(i), py(i) + R + .34, String(i), {c:COL.grey, fs:.26}));
  }
  return out;
}

/* -------------------------------------------------------------- renderer */
const svgNS = 'http://www.w3.org/2000/svg';
const mk = (tag, a) => { const e = document.createElementNS(svgNS, tag);
  for (const k in a) e.setAttribute(k, a[k]); return e; };
const clear = svg => { while (svg.firstChild) svg.removeChild(svg.firstChild); };

function defs(svg){
  const d = mk('defs', {});
  const f = mk('filter', {id:'glow', x:'-70%', y:'-70%', width:'240%', height:'240%'});
  f.appendChild(mk('feGaussianBlur', {stdDeviation:'.055', result:'b'}));
  const m = mk('feMerge', {});
  m.appendChild(mk('feMergeNode', {in:'b'}));
  m.appendChild(mk('feMergeNode', {in:'SourceGraphic'}));
  f.appendChild(m); d.appendChild(f);
  Object.keys(STY).forEach(k => {
    const mk2 = mk('marker', {id:'ar-' + k, viewBox:'0 0 10 10', refX:'8.5', refY:'5',
      markerWidth:'5.2', markerHeight:'5.2', orient:'auto-start-reverse'});
    mk2.appendChild(mk('path', {d:'M 0 1 L 9 5 L 0 9 z', fill:STY[k].stroke}));
    d.appendChild(mk2);
  });
  svg.appendChild(d);
}

function drawShape(svg, sh){
  const st = STY[sh.s || 'idle'];
  if (sh.t === 'e'){
    let {x1, y1, x2, y2} = sh;
    if (sh.pad){
      const dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy) || 1;
      x1 += dx / L * sh.pad; y1 += dy / L * sh.pad;
      x2 -= dx / L * sh.pad; y2 -= dy / L * sh.pad;
    }
    const a = {x1:x1.toFixed(3), y1:y1.toFixed(3), x2:x2.toFixed(3), y2:y2.toFixed(3),
      stroke:st.stroke, 'stroke-width':(sh.w || st.w || .05), 'stroke-linecap':'round',
      opacity:(sh.o == null ? (sh.s && sh.s !== 'idle' ? 1 : .75) : sh.o)};
    if (sh.dash || st.dash) a['stroke-dasharray'] = sh.dash || st.dash;
    if (sh.arrow !== false) a['marker-end'] = 'url(#ar-' + (sh.s || 'idle') + ')';
    svg.appendChild(mk('line', a));
    if (sh.lab != null)
      svg.appendChild(txt((x1 + x2) / 2 + (sh.lx || 0), (y1 + y2) / 2 + (sh.ly || -.16),
        tr(sh.lab), st.stroke, sh.fs || .30, 'middle'));
    return;
  }
  if (sh.t === 't'){
    svg.appendChild(txt(sh.x, sh.y, tr(sh.s), sh.c || COL.pale, sh.fs || .32,
      sh.anchor || 'middle', sh.o));
    return;
  }
  let node;
  if (sh.t === 'c'){
    node = mk('circle', {cx:sh.x.toFixed(3), cy:sh.y.toFixed(3), r:sh.r.toFixed(3),
      fill:st.fill, stroke:st.stroke, 'stroke-width':st.w});
  } else {
    node = mk('rect', {x:sh.x.toFixed(3), y:sh.y.toFixed(3), width:sh.w.toFixed(3),
      height:sh.h.toFixed(3), rx:(sh.rx == null ? .09 : sh.rx), fill:st.fill,
      stroke:st.stroke, 'stroke-width':st.w});
  }
  if (st.dash || sh.dash) node.setAttribute('stroke-dasharray', sh.dash || st.dash);
  if (st.glow) node.setAttribute('filter', 'url(#glow)');
  if (sh.o != null) node.setAttribute('opacity', sh.o);
  svg.appendChild(node);
  const cx = sh.t === 'c' ? sh.x : sh.x + sh.w / 2;
  const cy = sh.t === 'c' ? sh.y : sh.y + sh.h / 2;
  if (sh.lab != null && sh.lab !== '')
    svg.appendChild(txt(cx + (sh.dx || 0), cy + (sh.fs || .40) * .35, tr(sh.lab), st.text,
      sh.fs || .40, 'middle'));
  if (sh.sub != null)
    svg.appendChild(txt(cx, cy + (sh.t === 'c' ? sh.r : sh.h) + .34, tr(sh.sub),
      sh.subc || COL.grey, sh.subfs || .28, 'middle'));
  if (sh.top != null)
    svg.appendChild(txt(cx, cy - (sh.t === 'c' ? sh.r : sh.h / 2) - .22, tr(sh.top),
      sh.topc || COL.purpleL, sh.topfs || .28, 'middle'));
}
function txt(x, y, s, c, fs, anchor, o){
  const t = mk('text', {x:(+x).toFixed(3), y:(+y).toFixed(3), 'text-anchor':anchor || 'middle',
    fill:c, 'font-size':(+fs).toFixed(3)});
  if (o != null) t.setAttribute('opacity', o);
  t.textContent = s;
  return t;
}

function renderStage(frame){
  const svg = $('stage');
  clear(svg); defs(svg);
  const v = frame.view || curTab().view || [10, 6.4];
  svg.setAttribute('viewBox', '0 0 ' + v[0] + ' ' + v[1]);
  (frame.shapes || []).forEach(sh => drawShape(svg, sh));
}

function drawPanels(frame){
  const box = $('panels'); box.innerHTML = '';
  (frame.panels || []).forEach(p => {
    const d = document.createElement('div'); d.className = 'panel';
    const l = document.createElement('div'); l.className = 'lbl'; l.textContent = tr(p.lbl);
    const c = document.createElement('div'); c.className = 'chips';
    (p.chips.length ? p.chips : [{t:'—', cls:'empty'}]).forEach(ch => {
      const s = document.createElement('span');
      s.className = 'chip ' + (ch.cls || ''); s.textContent = ch.t; c.appendChild(s);
    });
    d.appendChild(l); d.appendChild(c); box.appendChild(d);
  });
}
function drawCode(frame){
  const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  $('code').innerHTML = curTab().code.map((ln, i) => {
    const h = esc(ln).replace(/(#.*)$/, '<span class="cm">$1</span>');
    return '<span class="ln' + (i === frame.line ? ' on' : '') + '">' + (h || ' ') + '</span>';
  }).join('');
}
function drawLegend(){
  const keys = curTab().legend || ['hot', 'act', 'ok', 'idle'];
  $('legend').innerHTML = keys.map(k => {
    const [c, txt] = Array.isArray(k) ? k : LEGEND[k];
    return '<span><i style="background:' + c + '"></i>' + tr(txt) + '</span>';
  }).join('');
}

let tabIx = 0, varIx = 0, frames = [], cur = 0, timer = null;
const TABS = DAY_META.tabs;
const curTab = () => TABS[tabIx];

function render(){
  if (!frames.length) return;
  cur = Math.max(0, Math.min(cur, frames.length - 1));
  const f = frames[cur];
  $('stage-title').textContent = tr(curTab().stage);
  renderStage(f); drawPanels(f); drawCode(f); drawLegend();
  $('narr').innerHTML = tr(f.msg);
  $('idea').innerHTML = tr(curTab().idea);
  $('stepno').textContent = (cur + 1) + ' / ' + frames.length;
  $('prev').disabled = cur === 0;
  $('next').disabled = cur === frames.length - 1;
}
function buildTabs(){
  $('tabs').innerHTML = '';
  TABS.forEach((t, i) => {
    const b = document.createElement('button');
    b.textContent = tr(t.label);
    if (i === tabIx) b.classList.add('on');
    b.onclick = () => { tabIx = i; varIx = 0; load(); };
    $('tabs').appendChild(b);
  });
  const ex = $('extra'); ex.innerHTML = '';
  const vs = curTab().variants;
  if (vs) vs.forEach((v, i) => {
    const b = document.createElement('button');
    b.textContent = tr(v);
    if (i === varIx) b.classList.add('on');
    b.onclick = () => { varIx = i; load(); };
    ex.appendChild(b);
  });
}
function load(){
  stop();
  frames = curTab().build(varIx) || [];
  cur = 0; buildTabs(); render();
}
function step(d){ stop(); cur += d; render(); }
function reset(){ stop(); cur = 0; render(); }
function stop(){ if (timer){ clearInterval(timer); timer = null; $('play').textContent = tr(T.play); } }
function togglePlay(){
  if (timer){ stop(); return; }
  if (cur >= frames.length - 1) cur = 0;
  $('play').textContent = tr(T.pause);
  timer = setInterval(() => {
    if (cur >= frames.length - 1){ stop(); return; }
    cur++; render();
  }, Number($('speed').value));
}
$('speed').addEventListener('input', () => { if (timer){ stop(); togglePlay(); } });
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowRight'){ step(1); e.preventDefault(); }
  else if (e.key === 'ArrowLeft'){ step(-1); e.preventDefault(); }
  else if (e.key === ' '){ togglePlay(); e.preventDefault(); }
});
setLang('zh');
load();
