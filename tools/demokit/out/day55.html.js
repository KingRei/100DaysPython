// DAY: 55
// TITLE_ZH: 從零訓一個 char-level mini-GPT：十萬個參數，學會一套文法
// TITLE_EN: A character-level mini-GPT from scratch: 100k parameters learning a grammar
// SUB_ZH: 前四天的東西全部湊在一個檔案裡：tokenizer、multi-head attention、transformer block、學出來的位置、Adam 訓練迴圈、取樣。沒有框架也沒有 autograd，反向傳播是手寫的，而語料來自一套已知的文法，所以「它學到東西了」是一個數字而不是感覺。
// SUB_EN: Everything from the last four days in one file - tokeniser, multi-head attention, a transformer block, learned positions, an Adam loop and sampling. No framework and no autograd; the backward pass is written by hand, and because the corpus comes from a known grammar, "it learned something" is a number rather than an impression.
// FOLDER: day%2055%20-%20char%20level%20mini%20gpt
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
const fmtN = x => x == null ? 'nan' : String(Math.round(x * 1e4) / 1e4);
const PURPLE = '#b45cff';

function barRow(sh, x0, base, bw, gap, vals, sts, labs, maxh, top){
  const hi = top || Math.max.apply(null, vals.map(Math.abs));
  vals.forEach((v, i) => {
    const hgt = Math.max(.05, maxh * Math.abs(v) / hi), x = x0 + i * (bw + gap);
    sh.push(S.r(x, base - hgt, bw, hgt, sts[i] || 'ok', null));
    if (labs && labs[i] != null) sh.push(fitT(x + bw / 2, base - hgt - .20, labs[i], {c:COL.pale, fs:.24}));
  });
  return sh;
}
// a 0..1 x 0..1 plot box; data y grows upward, SVG y grows downward
function Panel(x0, y0, w, h){ this.x0 = x0; this.y0 = y0; this.w = w; this.h = h; }
Panel.prototype.px = function(p){
  return [this.x0 + this.w * p[0], this.y0 + this.h * (1 - p[1])];
};
Panel.prototype.frame = function(sh){
  sh.push(S.r(this.x0, this.y0, this.w, this.h, 'ghost', null));
  return sh;
};
function polyline(sh, panel, pts, st, w){
  for (let i = 1; i < pts.length; i++){
    const a = panel.px(pts[i - 1]), b = panel.px(pts[i]);
    sh.push(S.e(a[0], a[1], b[0], b[1], {s:st, arrow:false, w:w || .055}));
  }
  return sh;
}
const SETUP = {"chars": 38005, "vocab": 24, "params": 104728, "ent": 0.326, "bits": 10.288, "avgLen": 31.56, "ctx": 32, "layers": 2, "heads": 4, "d": 64, "start": 3.1781, "sample": "a quiet bird slept in the garden.\nthe small cat sat near the door.\na bird sang near the door.\na ", "grad": 3.1e-06};
const TRAIN = {"hist": [3.1724, 3.0232, 2.9386, 2.8674, 2.7989, 2.7404, 2.6907, 2.6423, 2.5973, 2.5596, 2.5198, 2.4859, 2.4524, 2.4188, 2.3858, 2.3563, 2.3268, 2.2995, 2.2722, 2.244, 2.2191, 2.1502, 2.0943, 2.0414, 1.9934, 1.9518, 1.912, 1.8735, 1.8366, 1.8038, 1.7688, 1.7407, 1.7112, 1.6835, 1.659, 1.6353, 1.6129, 1.5931, 1.5732, 1.5514, 1.5336, 1.5132, 1.4972, 1.4789, 1.4604, 1.445, 1.4286, 1.4146, 1.3985, 1.3835, 1.3657, 1.3549, 1.3383, 1.3254, 1.312, 1.2942, 1.2788, 1.2642, 1.2465, 1.2277, 1.2176, 1.205, 1.1941, 1.1782, 1.1641, 1.1491, 1.132, 1.1169, 1.0978, 1.0837, 1.0714, 1.0592, 1.0458, 1.0329, 1.0212, 1.0068, 0.9978, 0.9867, 0.9756, 0.9638, 0.9548, 0.9406, 0.9269, 0.9183, 0.9068, 0.8961, 0.885, 0.8784, 0.8656, 0.8589, 0.8479, 0.8381, 0.8285, 0.8156, 0.8058, 0.7939, 0.7837, 0.7721, 0.763, 0.7547, 0.7464, 0.738, 0.7335, 0.7261, 0.714, 0.7071, 0.6977, 0.6937, 0.6843, 0.6791, 0.6744, 0.6691, 0.6612, 0.6527, 0.6473, 0.6412, 0.6372, 0.6333, 0.631, 0.6248, 0.6147, 0.6082, 0.6023, 0.5942, 0.5892, 0.584, 0.5774, 0.5705, 0.5648, 0.5594, 0.5551, 0.5481, 0.543, 0.5409, 0.5382, 0.5324, 0.5288, 0.5235, 0.5168, 0.5121, 0.5112, 0.5108, 0.5084, 0.5047, 0.4993, 0.4948, 0.491, 0.4889, 0.4882, 0.4877, 0.4838, 0.4804, 0.4761, 0.4733, 0.4693, 0.4679, 0.4659, 0.4631, 0.4596, 0.4582, 0.4566, 0.4504, 0.4444, 0.4444, 0.4431, 0.443, 0.4411, 0.441, 0.4387, 0.4364, 0.4311, 0.4297, 0.4297, 0.4263, 0.4248, 0.4231, 0.4211, 0.4225, 0.4191, 0.4157, 0.4142, 0.4115, 0.4126, 0.4111, 0.4101, 0.4113, 0.4109, 0.4113, 0.4087, 0.4068, 0.4074, 0.4085, 0.4073, 0.4053, 0.4055, 0.4037, 0.4017, 0.4032, 0.4018, 0.4019, 0.4029, 0.4026, 0.4019, 0.4008, 0.4035, 0.4016, 0.3985, 0.3963, 0.3947, 0.3946, 0.3955, 0.3963, 0.3936, 0.3911, 0.3887, 0.3898, 0.3903, 0.3908, 0.3874, 0.3849, 0.3839, 0.3864, 0.3844, 0.3834, 0.3824, 0.3839, 0.3821, 0.3829, 0.3827, 0.3827, 0.3831, 0.3839, 0.3818, 0.3824, 0.385, 0.387, 0.3853, 0.3864, 0.3864, 0.3839, 0.3832, 0.3844, 0.3787, 0.3816, 0.3809, 0.3806, 0.3794, 0.3793, 0.3771, 0.377, 0.3787, 0.3797, 0.3782, 0.3798, 0.3808, 0.3805, 0.3781, 0.3777, 0.3756, 0.3743, 0.3757, 0.3752, 0.3754, 0.3757, 0.3736, 0.3725, 0.3714, 0.3679, 0.3664, 0.3646, 0.3636, 0.362, 0.3581, 0.3545, 0.3505, 0.3477, 0.3454, 0.3443, 0.344, 0.346, 0.3461, 0.3463, 0.3467, 0.3448, 0.3444, 0.3436, 0.3426, 0.344, 0.3461, 0.3474, 0.3498, 0.3502, 0.3485, 0.3499, 0.3542, 0.3555, 0.3557, 0.3572, 0.3603, 0.3604, 0.3599, 0.3603, 0.3613, 0.3621, 0.3621, 0.3644, 0.366, 0.3677, 0.3655, 0.3649, 0.3642, 0.362, 0.3623, 0.3642, 0.3639, 0.3625, 0.3633, 0.366, 0.3655, 0.3677, 0.3667, 0.3677, 0.3669, 0.3657, 0.3649, 0.3643, 0.362, 0.3628, 0.3625, 0.3644, 0.3641, 0.3656, 0.3659, 0.3654, 0.3661, 0.3661, 0.3641, 0.3637, 0.3622, 0.3618, 0.3586, 0.3582, 0.3545, 0.355, 0.3536, 0.3532, 0.3528, 0.354, 0.352, 0.3509, 0.3514, 0.3518, 0.3515, 0.3518, 0.3514, 0.3483, 0.3485, 0.3478, 0.3464, 0.3456, 0.3452, 0.3437, 0.3439, 0.3439, 0.3435, 0.3447, 0.3427, 0.341, 0.3384, 0.337, 0.3355, 0.3328, 0.3306, 0.3288, 0.3284, 0.3263, 0.3291, 0.3281, 0.3292, 0.3276, 0.3267, 0.3272, 0.3263, 0.328, 0.3295, 0.3298, 0.3301, 0.3312, 0.3338, 0.3353, 0.3385, 0.3398, 0.3413, 0.3417, 0.3441, 0.3471, 0.3486, 0.3452, 0.3466, 0.3479, 0.3514, 0.3528, 0.3513, 0.353, 0.3528, 0.3509, 0.3502, 0.3503, 0.3499, 0.3507, 0.3495, 0.3495, 0.3529, 0.3539, 0.3558, 0.3556, 0.3539, 0.3549, 0.3575, 0.3588, 0.3574, 0.356, 0.3554, 0.3578, 0.3578, 0.3572, 0.3583, 0.3585, 0.3579, 0.3582, 0.357, 0.3578, 0.3556, 0.3525, 0.3515, 0.3499, 0.3484, 0.3466, 0.3453, 0.343, 0.3406, 0.3402, 0.3396, 0.3394, 0.3367, 0.3353, 0.3334, 0.3314, 0.3295, 0.3281, 0.3277, 0.3271, 0.3278, 0.3268, 0.3262, 0.324, 0.3231, 0.3254, 0.3271, 0.3284, 0.3291, 0.329, 0.3294, 0.3302, 0.3307, 0.3318, 0.3325, 0.3346, 0.3367, 0.3389, 0.3399, 0.3405, 0.3439, 0.3448, 0.3454, 0.3457, 0.3474, 0.3492, 0.3485, 0.3481, 0.3469, 0.3476, 0.3477, 0.347, 0.3462, 0.3447, 0.3446, 0.3442, 0.3436, 0.3428, 0.34, 0.3385, 0.3374, 0.3353, 0.3349, 0.3335, 0.3323, 0.3312, 0.3308, 0.329], "nomask": [3.1777, 3.0244, 2.9434, 2.8703, 2.8, 2.7385, 2.6877, 2.6379, 2.594, 2.5551, 2.5145, 2.48, 2.4469, 2.4132, 2.3804, 2.3519, 2.3224, 2.2964, 2.2696, 2.2425, 2.2183, 2.1509, 2.0957, 2.0452, 1.9988, 1.958, 1.9209, 1.8868, 1.8529, 1.8228, 1.7928, 1.7716, 1.746, 1.7221, 1.7011, 1.6839, 1.6647, 1.6519, 1.638, 1.6243, 1.6127, 1.6001, 1.5885, 1.5783, 1.5649, 1.5547, 1.5462, 1.5394, 1.5293, 1.5228, 1.5139, 1.5067, 1.4958, 1.4905, 1.4862, 1.48, 1.4721, 1.4679, 1.4576, 1.4503, 1.4456, 1.4433, 1.4394, 1.4353, 1.4338, 1.4328, 1.4303, 1.4277, 1.4181, 1.4146, 1.4118, 1.4093, 1.4074, 1.403, 1.4001, 1.3945, 1.3931, 1.3906, 1.3834, 1.3805, 1.3743, 1.3694, 1.3657, 1.3637, 1.3569, 1.3503, 1.3442, 1.3402, 1.3334, 1.3302, 1.3315, 1.3263, 1.3205, 1.3154, 1.3113, 1.3067, 1.3027, 1.2968, 1.2935, 1.2926, 1.2899, 1.289, 1.289, 1.2843, 1.2819, 1.2802, 1.2751, 1.2734, 1.2697, 1.2671, 1.267, 1.2579, 1.2539, 1.2519, 1.2506, 1.2471, 1.244, 1.2392, 1.236, 1.2286, 1.2207, 1.2145, 1.2075, 1.1978, 1.1909, 1.1792, 1.1698, 1.1578, 1.146, 1.1322, 1.1152, 1.0997, 1.0831, 1.0649, 1.0463, 1.0251, 1.0049, 0.9864, 0.9665, 0.9447, 0.9261, 0.9036, 0.8791, 0.8563, 0.8317, 0.8041, 0.7776, 0.7517, 0.7319, 0.7069, 0.6808, 0.658, 0.6344, 0.6128, 0.5902, 0.5671, 0.5441, 0.5209, 0.4951, 0.4725, 0.4502, 0.4262, 0.4056, 0.386, 0.3656, 0.3465, 0.3284, 0.3116, 0.2922, 0.271, 0.2543, 0.239, 0.2253, 0.2111, 0.1951, 0.1838, 0.1725, 0.1613, 0.1534, 0.142, 0.134, 0.1254, 0.1182, 0.1118, 0.106, 0.0993, 0.0939, 0.0891, 0.0861, 0.0845, 0.0824, 0.0802, 0.0793, 0.0792, 0.0763, 0.0764, 0.0756, 0.0744, 0.0731, 0.0715, 0.0726, 0.07, 0.0683, 0.0683, 0.0676, 0.0682, 0.067, 0.0682, 0.0676, 0.067, 0.0656, 0.0647, 0.0633, 0.0621, 0.0589, 0.059, 0.0574, 0.055, 0.0535, 0.0531, 0.0515, 0.0511, 0.052, 0.0531, 0.0528, 0.0535, 0.0518, 0.0523, 0.0505, 0.0507, 0.0497, 0.0489, 0.0492, 0.0485, 0.0484, 0.0492, 0.0481, 0.0473, 0.0466, 0.0478, 0.0474, 0.0482, 0.0463, 0.0459, 0.0445, 0.0441, 0.0434, 0.0429, 0.0423, 0.0432], "final": 0.3375, "val": 0.3301, "finalBits": 0.4869, "valBits": 0.4762, "noMaskLoss": 0.0491, "maskedAt250": 0.3824, "shiftLoss": 0.0012, "shiftRef": 0.4863};
const SAMPLES = [{"temp": 1.0, "topk": null, "frac": 67.0, "n": 6, "lines": [{"t": "the bird ran under the tree.", "ok": true}, {"t": "a quiet bird sathe garden.", "ok": false}, {"t": "the old cat ran under the tree.", "ok": true}, {"t": "the bird ran the garden.", "ok": false}]}, {"temp": 0.8, "topk": 5, "frac": 50.0, "n": 6, "lines": [{"t": "the quiet child ran near the doog ran under the tree", "ok": false}, {"t": "the old bird waithed in the garden.", "ok": false}, {"t": "the old dog sang in the garden.", "ok": true}, {"t": "a old child waithed in the garden.", "ok": false}]}, {"temp": 0.2, "topk": null, "frac": 100.0, "n": 7, "lines": [{"t": "the old robot sang under the tree.", "ok": true}, {"t": "the old robot slept in the garden.", "ok": true}, {"t": "a old robot ran in the garden.", "ok": true}, {"t": "a small bird slept in the garden.", "ok": true}]}];
const BAD = {"frac": 0.0, "n": 1, "lines": [" de dededededededededededededededededededededededede", "a dedededededededededededededdedededededededededede"], "shift": "the                                                 "};
const ATT = {"text": "the small cat sat on the mat.|th", "w": [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.052, 0.948, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.027, 0.768, 0.205, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.369, 0.239, 0.175, 0.216, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.253, 0.164, 0.051, 0.101, 0.432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.138, 0.097, 0.071, 0.138, 0.333, 0.223, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.084, 0.172, 0.089, 0.033, 0.06, 0.379, 0.183, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.021, 0.188, 0.023, 0.002, 0.015, 0.185, 0.498, 0.068, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.015, 0.216, 0.036, 0.003, 0.013, 0.098, 0.501, 0.081, 0.038, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.08, 0.132, 0.142, 0.054, 0.05, 0.191, 0.089, 0.08, 0.105, 0.076, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.04, 0.092, 0.213, 0.221, 0.109, 0.03, 0.077, 0.092, 0.058, 0.018, 0.051, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.081, 0.065, 0.074, 0.09, 0.048, 0.114, 0.041, 0.058, 0.104, 0.264, 0.031, 0.031, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.013, 0.109, 0.023, 0.005, 0.016, 0.03, 0.271, 0.062, 0.026, 0.006, 0.027, 0.396, 0.014, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.086, 0.018, 0.045, 0.259, 0.08, 0.032, 0.01, 0.027, 0.046, 0.208, 0.026, 0.008, 0.093, 0.063, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.009, 0.012, 0.01, 0.02, 0.026, 0.009, 0.022, 0.012, 0.007, 0.005, 0.028, 0.053, 0.057, 0.658, 0.072, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.028, 0.042, 0.011, 0.006, 0.011, 0.062, 0.051, 0.032, 0.049, 0.116, 0.015, 0.019, 0.002, 0.002, 0.018, 0.534, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.005, 0.042, 0.008, 0.002, 0.007, 0.01, 0.126, 0.024, 0.008, 0.002, 0.014, 0.209, 0.007, 0.047, 0.065, 0.413, 0.011, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.008, 0.001, 0.001, 0.002, 0.005, 0.005, 0.002, 0.002, 0.003, 0.025, 0.007, 0.001, 0.001, 0.01, 0.013, 0.007, 0.002, 0.905, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.005, 0.002, 0.001, 0.002, 0.002, 0.004, 0.003, 0.004, 0.006, 0.03, 0.002, 0.001, 0.0, 0.0, 0.003, 0.02, 0.0, 0.876, 0.039, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.048, 0.0, 0.882, 0.065, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.001, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.004, 0.001, 0.0, 0.0, 0.001, 0.002, 0.003, 0.0, 0.89, 0.006, 0.0, 0.09, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.025, 0.06, 0.211, 0.154, 0.03, 0.013, 0.024, 0.069, 0.071, 0.035, 0.007, 0.036, 0.062, 0.006, 0.003, 0.008, 0.036, 0.0, 0.005, 0.138, 0.001, 0.007, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.009, 0.092, 0.06, 0.004, 0.006, 0.026, 0.077, 0.044, 0.03, 0.003, 0.012, 0.131, 0.009, 0.002, 0.01, 0.192, 0.009, 0.011, 0.059, 0.014, 0.066, 0.006, 0.128, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.003, 0.041, 0.011, 0.001, 0.003, 0.006, 0.078, 0.024, 0.01, 0.001, 0.005, 0.113, 0.002, 0.002, 0.009, 0.267, 0.003, 0.017, 0.144, 0.003, 0.089, 0.009, 0.147, 0.009, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.002, 0.003, 0.0, 0.895, 0.006, 0.0, 0.088, 0.001, 0.001, 0.0, 0.002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.028, 0.019, 0.035, 0.104, 0.057, 0.013, 0.018, 0.028, 0.025, 0.03, 0.025, 0.024, 0.096, 0.19, 0.029, 0.005, 0.082, 0.002, 0.013, 0.047, 0.002, 0.053, 0.011, 0.025, 0.017, 0.021, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.041, 0.027, 0.11, 0.164, 0.028, 0.042, 0.008, 0.026, 0.063, 0.147, 0.01, 0.01, 0.035, 0.004, 0.002, 0.003, 0.02, 0.002, 0.002, 0.124, 0.003, 0.003, 0.006, 0.065, 0.03, 0.009, 0.018, 0.0, 0.0, 0.0, 0.0, 0.0], [0.017, 0.067, 0.131, 0.045, 0.023, 0.012, 0.046, 0.069, 0.045, 0.009, 0.013, 0.077, 0.055, 0.01, 0.008, 0.02, 0.041, 0.0, 0.015, 0.057, 0.002, 0.019, 0.046, 0.053, 0.007, 0.01, 0.077, 0.024, 0.0, 0.0, 0.0, 0.0], [0.033, 0.004, 0.013, 0.156, 0.06, 0.007, 0.004, 0.012, 0.015, 0.062, 0.025, 0.003, 0.071, 0.14, 0.018, 0.001, 0.056, 0.002, 0.003, 0.048, 0.001, 0.068, 0.003, 0.017, 0.022, 0.014, 0.006, 0.056, 0.079, 0.0, 0.0, 0.0], [0.035, 0.025, 0.041, 0.062, 0.041, 0.05, 0.018, 0.021, 0.029, 0.042, 0.034, 0.029, 0.065, 0.077, 0.028, 0.01, 0.056, 0.014, 0.011, 0.039, 0.009, 0.018, 0.009, 0.028, 0.023, 0.037, 0.032, 0.045, 0.042, 0.031, 0.0, 0.0], [0.015, 0.056, 0.188, 0.065, 0.019, 0.015, 0.024, 0.045, 0.042, 0.01, 0.01, 0.067, 0.067, 0.007, 0.004, 0.007, 0.041, 0.0, 0.004, 0.063, 0.001, 0.008, 0.016, 0.052, 0.004, 0.007, 0.08, 0.028, 0.034, 0.015, 0.008, 0.0], [0.004, 0.052, 0.031, 0.002, 0.004, 0.019, 0.053, 0.019, 0.011, 0.001, 0.013, 0.158, 0.009, 0.003, 0.014, 0.104, 0.01, 0.005, 0.038, 0.005, 0.024, 0.007, 0.057, 0.015, 0.002, 0.016, 0.092, 0.007, 0.037, 0.074, 0.004, 0.112]]};


/* ================================================================ tab 1: the setup */
const CODE_SETUP = [
  'DET   = ["the ", "a "]',
  'ADJ   = ["small ", "old ", "red ", "quiet ", ""]',
  'NOUN  = ["cat", "dog", "bird", "child", "robot"]',
  'VERB  = ["sat", "slept", "waited", "sang", "ran"]',
  'PLACE = ["on the mat", "in the garden", ...]',
  '',
  '# five independent choices per sentence = a known entropy'
];
function setupStage(o){
  const sh = [];
  const slots = [{n:'DET', k:2}, {n:'ADJ', k:5}, {n:'NOUN', k:5},
                 {n:'VERB', k:5}, {n:'PLACE', k:5}];
  slots.forEach((s, i) => {
    if (i > o.upto) return;
    const x = .75 + i * 1.82;
    sh.push(box(x, 3.70, 1.62, .62, i === o.upto ? 'hot' : 'ok', s.n, .30));
    sh.push(fitT(x + .81, 3.42, {zh:s.k + ' 種選擇', en:s.k + ' choices'},
                 {c:COL.grey, fs:.24}));
  });
  if (o.upto >= 4){
    sh.push(fitT(4.9, 2.70, SETUP.bits.toFixed(3) + ' bits / ' + SETUP.avgLen.toFixed(2) +
                 ' chars = ' + SETUP.ent.toFixed(4) + ' bits/char',
                 {c:COL.tealL, fs:.34}));
  }
  side(sh, .75, 1.85, o.notes || [], .46);
  return caps(sh, o);
}
function setupFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:setupStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:1, line:1,
    cap:{zh:'語料是自己長出來的，不是抓來的', en:'the corpus is generated, not scraped'},
    panels:[{lbl:'characters', chips:[chip(SETUP.chars.toLocaleString(), 'ok')]},
            {lbl:'vocabulary', chips:[chip(String(SETUP.vocab), 'ok')]}],
    msg:{zh:'每個句子是五個獨立的選擇填進五個空格，像「' + SETUP.sample.split('\n')[0] + '」。整份語料 ' + SETUP.chars.toLocaleString() + ' 個字元、只有 ' + SETUP.vocab + ' 種不同字元——所以今天的 tokenizer 就是「一個字元一個 token」，vocabulary 只有 ' + SETUP.vocab + ' 個。',
         en:'Each sentence fills five slots with five independent choices, like "' + SETUP.sample.split('\n')[0] + '". The whole corpus is ' + SETUP.chars.toLocaleString() + ' characters drawn from just ' + SETUP.vocab + ' distinct ones, so today\'s tokeniser is simply "one character, one token" and the vocabulary has ' + SETUP.vocab + ' entries.'}});
  p({upto:4, line:6,
    cap:{zh:'而這買到一條別人沒有的底線', en:'and that buys a floor nobody else has'},
    panels:[{lbl:'per sentence', chips:[chip(SETUP.bits.toFixed(3) + ' bits', 'hot')]},
            {lbl:'per character', chips:[chip(SETUP.ent.toFixed(4) + ' bits', 'hot')]}],
    msg:{zh:'因為產生器是已知的，這份文字的真正熵也是已知的：一個句子的資訊量是五個選擇的 log2 相加＝' + SETUP.bits.toFixed(3) + ' bits，平均攤在 ' + SETUP.avgLen.toFixed(2) + ' 個字元上就是 ' + SETUP.ent.toFixed(4) + ' bits/char。沒有模型能低於它；低很多就是在背答案。抓來的網路文字給不了你這條線，所以「loss 降了」永遠只能跟自己比。',
         en:'Because the generator is known, so is the true entropy of the text: a sentence is worth the sum of five log2 counts, ' + SETUP.bits.toFixed(3) + ' bits, spread over ' + SETUP.avgLen.toFixed(2) + ' characters on average - ' + SETUP.ent.toFixed(4) + ' bits per character. No model can go below it, and one that goes far below is memorising. Scraped text gives you no such line, which is why "the loss went down" can only ever be compared with itself.'}});
  p({upto:4, line:0,
    cap:{zh:'模型本身很小', en:'the model itself is tiny'},
    panels:[{lbl:'parameters', chips:[chip(SETUP.params.toLocaleString(), 'act')]},
            {lbl:'shape', chips:[chip('d ' + SETUP.d, 'dim'), chip(SETUP.layers + ' blocks', 'dim'),
                                 chip(SETUP.heads + ' heads', 'dim'), chip('ctx ' + SETUP.ctx, 'dim')]}],
    msg:{zh:'d_model ' + SETUP.d + '、' + SETUP.layers + ' 個 block、' + SETUP.heads + ' 個 head、context ' + SETUP.ctx + '，總共 ' + SETUP.params.toLocaleString() + ' 個參數——跟一張照片差不多大。零件就是前四天那些：字元 id →  embedding ＋ 學出來的位置 → pre-LN 的 (attention, MLP) → 最後 norm → 每個字元一個分數。反向傳播是手寫的，跟中央差分對過，最差相對誤差 ' + SETUP.grad.toExponential(0) + '。',
         en:'d_model ' + SETUP.d + ', ' + SETUP.layers + ' blocks, ' + SETUP.heads + ' heads, context ' + SETUP.ctx + ', ' + SETUP.params.toLocaleString() + ' parameters - about the size of a photograph. The pieces are the last four days: character ids to embedding plus a learned position, pre-LN blocks of attention and MLP, a final norm, one score per character. The backward pass is written by hand and agrees with central differences to ' + SETUP.grad.toExponential(0) + '.'}});
  return F.list;
}

/* ================================================================ tab 2: training */
const CODE_TRAIN = [
  'x = data[i : i + ctx]          # the characters',
  'y = data[i + 1 : i + ctx + 1]  # the same, shifted by one',
  '',
  'logits, cache = forward(p, cfg, x)',
  'loss, dlogits = cross_entropy(logits, y)',
  'grads = backward(p, cfg, cache, dlogits)',
  '# ... then one Adam step per parameter tensor'
];
function curveStage(o){
  const sh = [], X0 = 1.45, Y0 = 1.60, WD = 7.0, HT = 2.60;
  const top = SETUP.start * 1.05;
  sh.push(S.e(X0, Y0, X0 + WD, Y0, {s:'ghost', arrow:false, w:.04}));
  sh.push(S.e(X0, Y0, X0, Y0 + HT, {s:'ghost', arrow:false, w:.04}));
  const draw = (vals, st, upto) => {
    for (let i = 1; i < vals.length; i++){
      if (upto != null && i > upto) break;
      sh.push(S.e(X0 + WD * (i - 1) / (TRAIN.hist.length - 1), Y0 + HT * vals[i - 1] / top,
                  X0 + WD * i / (TRAIN.hist.length - 1), Y0 + HT * vals[i] / top,
                  {s:st, arrow:false, w:.06}));
    }
  };
  draw(TRAIN.hist, 'ok', o.upto);
  if (o.nomask) draw(TRAIN.nomask, 'bad', o.upto);
  const floor = SETUP.ent * Math.LN2;
  sh.push(S.e(X0, Y0 + HT * floor / top, X0 + WD, Y0 + HT * floor / top,
              {s:'hot', arrow:false, w:.04}));
  sh.push(fitT(X0 + .12, Y0 + HT * floor / top - .26,
               {zh:'產生器的熵 ' + SETUP.ent.toFixed(4) + ' bits/char',
                en:'the generator\'s entropy: ' + SETUP.ent.toFixed(4) + ' bits/char'},
               {anchor:'start', c:COL.hot, fs:.26}));
  [[0, '0'], [Math.LN2, '1 bit'], [SETUP.start, 'ln(' + SETUP.vocab + ')']].forEach(v => {
    sh.push(fitT(X0 - .18, Y0 + HT * v[0] / top, v[1], {anchor:'end', c:COL.grey, fs:.24}));
  });
  sh.push(fitT(X0 + WD / 2, Y0 - .60, {zh:'訓練步數', en:'training step'}, {c:COL.grey, fs:.26}));
  return caps(sh, o);
}
function trainFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:curveStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:1, line:1,
    cap:{zh:'要學的就是「下一個字元」', en:'the task is the next character'},
    panels:[{lbl:'start', chips:[chip(SETUP.start.toFixed(4), 'dim')]}],
    msg:{zh:'輸入是 ' + SETUP.ctx + ' 個字元，目標是同樣那串往右移一格。一開始模型什麼都不知道，loss 應該是 ln(' + SETUP.vocab + ') = ' + SETUP.start.toFixed(4) + '——實測第一步 ' + TRAIN.hist[0].toFixed(4) + '，這是第一個該對的數字。',
         en:'The input is ' + SETUP.ctx + ' characters and the target is the same string shifted one place right. Knowing nothing costs ln(' + SETUP.vocab + ') = ' + SETUP.start.toFixed(4) + ', and the first step measures ' + TRAIN.hist[0].toFixed(4) + ' - the first number that should come out right.'}});
  [100, 250, 499].forEach(k => {
    p({upto:k, line:5,
      cap:{zh:'500 步、batch 16、Adam，沒有排程也沒有 clipping',
           en:'500 steps, batch 16, Adam, no schedule and no clipping'},
      panels:[{lbl:'step', chips:[chip(String(k + 1), 'hot')]},
              {lbl:'loss / bits', chips:[chip(TRAIN.hist[k].toFixed(4), 'act'),
                                         chip((TRAIN.hist[k] / Math.LN2).toFixed(3), 'act')]}],
      msg:k === 499
        ? {zh:'收在 ' + TRAIN.final.toFixed(4) + '＝' + TRAIN.finalBits.toFixed(4) + ' bits/char，而產生器的底線是 ' + SETUP.ent.toFixed(4) + '，差 ' + (TRAIN.finalBits - SETUP.ent).toFixed(4) + '。換一個 seed 產生的語料（模型沒看過）是 ' + TRAIN.valBits.toFixed(4) + ' bits/char，跟訓練集幾乎一樣——代表它學到的是文法，不是那 38,005 個字元。',
           en:'It settles at ' + TRAIN.final.toFixed(4) + ' = ' + TRAIN.finalBits.toFixed(4) + ' bits per character against a floor of ' + SETUP.ent.toFixed(4) + ', a gap of ' + (TRAIN.finalBits - SETUP.ent).toFixed(4) + '. On text from the same grammar with a different seed it scores ' + TRAIN.valBits.toFixed(4) + ' bits per character, essentially the same - so what it learned is the grammar, not the particular 38,005 characters.'}
        : {zh:'第 ' + (k + 1) + ' 步：' + TRAIN.hist[k].toFixed(4) + '（' + (TRAIN.hist[k] / Math.LN2).toFixed(3) + ' bits/char）。大部分的下降在前一百步就發生了，因為最容易學的是「字元的分布」和「句子會以句點加換行結束」。',
           en:'Step ' + (k + 1) + ': ' + TRAIN.hist[k].toFixed(4) + ' (' + (TRAIN.hist[k] / Math.LN2).toFixed(3) + ' bits per character). Most of the drop happens in the first hundred steps, because the easiest things to learn are which characters are common and that a sentence ends with a full stop and a newline.'}});
  });
  return F.list;
}

/* ================================================================ tab 3: sampling */
const CODE_SAMPLE = [
  'window = ids[-ctx:]',
  'logits, _ = forward(p, cfg, window)',
  'z = logits[0, -1] / temperature',
  'if top_k: keep only the k largest, set the rest to -inf',
  'ids.append(choice(len(probs), p=softmax(z)))',
  '',
  '# then do it again, with the new character appended'
];
function sampleStage(o){
  const sh = [], s = SAMPLES[o.which];
  const base = 2.45, maxh = 1.85;
  SAMPLES.forEach((r, i) => {
    const x = 1.25 + i * 2.65, h = Math.max(.05, maxh * r.frac / 100);
    const st = i === o.which ? 'hot' : (r.frac >= 90 ? 'ok' : 'idle');
    sh.push(S.r(x, base, 1.45, h, st, null));
    sh.push(fitT(x + .72, base + h + .26, r.frac.toFixed(0) + '%',
                 {c:i === o.which ? COL.hot : COL.tealL, fs:.32}));
    sh.push(fitT(x + .72, base - .30,
                 'temp ' + r.temp + (r.topk ? ', top-k ' + r.topk : ''),
                 {c:COL.grey, fs:.26}));
  });
  s.lines.forEach((l, i) => {
    sh.push(fitT(.75, 1.75 - i * .42, l.t, {anchor:'start', c:l.ok ? COL.tealL : COL.red, fs:.26}));
  });
  sh.push(fitT(4.9, 5.00, {zh:'產生的句子裡，文法合格的比例', en:'how many generated lines the grammar could have made'},
               {c:COL.grey, fs:.26}));
  return caps(sh, o);
}
function sampleFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:sampleStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  const msgs = [
    {zh:'temperature 1.0 就是照模型自己的機率抽。' + SAMPLES[0].frac.toFixed(0) + '% 的完整句子是文法產得出來的；錯的那些通常是在中間漏掉一段（像「sathe garden」），也就是一個字元抽歪就回不來了。',
     en:'At temperature 1.0 you sample straight from the model\'s own probabilities. ' + SAMPLES[0].frac.toFixed(0) + '% of complete lines are ones the grammar could have produced; the failures are usually a swallowed chunk in the middle, because one unlucky character cannot be taken back.'},
    {zh:'temperature 0.8 加 top-k 5 這次反而更差（' + SAMPLES[1].frac.toFixed(0) + '%）。這不是矛盾：樣本只有 ' + SAMPLES[1].n + ' 句，而一句裡只要一個字元歪掉整句就不合格——**這種小樣本的百分比本來就會跳**，要拿它比較就得抽幾百句。',
     en:'Temperature 0.8 with top-k 5 actually scores lower here (' + SAMPLES[1].frac.toFixed(0) + '%). That is not a contradiction: the sample is ' + SAMPLES[1].n + ' lines, and a single wrong character spoils a whole line, so a percentage from a handful of lines bounces around. Comparing settings properly means sampling hundreds.'},
    {zh:'temperature 0.2 幾乎是 greedy：' + SAMPLES[2].frac.toFixed(0) + '% 合格，但句子會開始重複（同樣的 robot 一直出現）。這就是取樣的那個取捨——調低溫度買到的是「更安全」，付出的是多樣性。',
     en:'Temperature 0.2 is nearly greedy: ' + SAMPLES[2].frac.toFixed(0) + '% grammatical, but the lines start repeating themselves (the same robot keeps turning up). That is the sampling trade in one picture: lowering the temperature buys safety and pays for it in variety.'}];
  SAMPLES.forEach((s, i) => {
    p({which:i, line:2,
      cap:{zh:'一次一個字元，把抽到的接回去再跑一次', en:'one character at a time, appended and fed back in'},
      panels:[{lbl:'setting', chips:[chip('temp ' + s.temp, 'hot'),
                                     chip(s.topk ? 'top-k ' + s.topk : 'no top-k', 'dim')]},
              {lbl:'grammatical', chips:[chip(s.frac.toFixed(0) + '% of ' + s.n, s.frac >= 90 ? 'ok' : 'dim')]}],
      msg:msgs[i]});
  });
  p({which:2, line:6,
    cap:{zh:'為什麼要用「文法合格率」而不是「看起來像不像」', en:'why a grammar check instead of "does it look plausible"'},
    panels:[{lbl:'check', chips:[chip('runnable', 'ok')]}],
    msg:{zh:'因為這是程式跑得出來的：把生成的每一行丟回五個空格的產生器看它產不產得出來。生成文字最容易自欺的地方就是「看起來很像」，而當語料是自己造的時候，就有一個不靠感覺的判準。',
         en:'Because it is something the file can actually run: take each generated line and ask whether the five-slot generator could have produced it. The easiest way to fool yourself about generated text is that it looks about right, and a generated corpus gives you a test that does not depend on how it looks.'}});
  return F.list;
}

/* ================================================================ tab 4: the two bugs */
const CODE_BUG = [
  '# bug 1: no causal mask - every position sees its own answer',
  'att = q @ k.T / sqrt(dh)        # (no where(mask, att, -inf))',
  '',
  '# bug 2: targets shifted by 0 - predict the character you were given',
  'y = data[i + 0 : i + ctx + 0]',
  '',
  '# both make the loss go DOWN'
];
function bugStage(o){
  const sh = [], base = 2.60, maxh = 2.00;
  const bars = o.mode === 'mask'
    ? [{v:TRAIN.maskedAt250, t:{zh:'有 mask（250 步）', en:'with the mask, 250 steps'}, st:'ok'},
       {v:TRAIN.noMaskLoss, t:{zh:'沒有 mask（250 步）', en:'no mask, 250 steps'}, st:'bad'}]
    : [{v:TRAIN.shiftRef, t:{zh:'目標往右移 1（150 步）', en:'targets shifted by 1, 150 steps'}, st:'ok'},
       {v:TRAIN.shiftLoss, t:{zh:'目標移 0（150 步）', en:'targets shifted by 0, 150 steps'}, st:'bad'}];
  const top = Math.max.apply(null, bars.map(b => b.v)) * 1.1;
  bars.forEach((b, i) => {
    const x = 1.75 + i * 3.10, h = Math.max(.05, maxh * b.v / top);
    sh.push(S.r(x, base, 1.85, h, b.st, null));
    sh.push(fitT(x + .92, base + h + .26, b.v.toFixed(4),
                 {c:b.st === 'ok' ? COL.tealL : COL.red, fs:.32}));
    sh.push(fitT(x + .92, base - .32, b.t, {c:COL.grey, fs:.26}));
  });
  (o.lines || []).forEach((l, i) => {
    sh.push(fitT(.75, 1.85 - i * .40, l, {anchor:'start', c:COL.red, fs:.26}));
  });
  sh.push(fitT(4.9, 5.00, {zh:'訓練 loss（越低越好⋯⋯對吧）', en:'training loss (lower is better, surely)'},
               {c:COL.grey, fs:.26}));
  return caps(sh, o);
}
function bugFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:bugStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({mode:'mask', line:1,
    cap:{zh:'第一個：訓練時忘了 causal mask', en:'bug one: training without the causal mask'},
    panels:[{lbl:'with mask', chips:[chip(TRAIN.maskedAt250.toFixed(4), 'ok')]},
            {lbl:'without', chips:[chip(TRAIN.noMaskLoss.toFixed(4), 'bad')]}],
    msg:{zh:'拿掉 mask 之後，同樣 250 步的 loss 從 ' + TRAIN.maskedAt250.toFixed(4) + ' 掉到 ' + TRAIN.noMaskLoss.toFixed(4) + '——看起來好非常多。它在作弊：每個位置都看得到自己要預測的那個字元。形狀沒錯、梯度沒錯、曲線漂亮，沒有任何東西會抗議。',
         en:'Drop the mask and the same 250 steps reach ' + TRAIN.noMaskLoss.toFixed(4) + ' instead of ' + TRAIN.maskedAt250.toFixed(4) + ' - far better, apparently. It is cheating: every position can see the character it is being asked to predict. The shapes are right, the gradients are right, the curve looks great, and nothing complains.'}});
  p({mode:'mask', line:1, lines:BAD.lines,
    cap:{zh:'只有在「生成」的時候才會露出來', en:'it only shows up when you generate'},
    panels:[{lbl:'grammatical', chips:[chip(BAD.frac.toFixed(0) + '%', 'bad')]},
            {lbl:'loss', chips:[chip(TRAIN.noMaskLoss.toFixed(4), 'bad')]}],
    msg:{zh:'讓這個「loss 比較低」的模型自己寫，' + BAD.n + ' 個完整句子裡 ' + BAD.frac.toFixed(0) + '% 合格——它只會吐重複的字母。因為生成的時候沒有未來可以偷看，而它學到的每一件事都建立在偷看上。**訓練 loss 和你真正要的東西之間，可以差到這個程度。**',
         en:'Ask the model with the lower loss to write, and ' + BAD.frac.toFixed(0) + '% of its ' + BAD.n + ' complete lines are grammatical - it emits repeated letters. At generation time there is no future to peek at, and everything it learned was built on peeking. That is how far apart the training loss and the thing you actually want can be.'}});
  p({mode:'shift', line:4,
    cap:{zh:'第二個：目標忘了往右移一格', en:'bug two: targets that were never shifted'},
    panels:[{lbl:'shift 1', chips:[chip(TRAIN.shiftRef.toFixed(4), 'ok')]},
            {lbl:'shift 0', chips:[chip(TRAIN.shiftLoss.toFixed(4), 'bad')]}],
    msg:{zh:'把目標設成「現在這個字元」而不是「下一個字元」，150 步的 loss 是 ' + TRAIN.shiftLoss.toFixed(4) + '（正確版是 ' + TRAIN.shiftRef.toFixed(4) + '）。loss 直奔 0 很容易被讀成成功，但它學的是恆等函數：把輸入抄過去。',
         en:'Make the target "the current character" instead of "the next character" and 150 steps land at ' + TRAIN.shiftLoss.toFixed(4) + ', against ' + TRAIN.shiftRef.toFixed(4) + ' done properly. A loss heading for zero is easy to read as success, but what it is learning is the identity function: copy the input through.'}});
  p({mode:'shift', line:6, lines:[BAD.shift],
    cap:{zh:'生成時要抄的那個字元剛好還不存在', en:'at generation time the character to copy does not exist yet'},
    panels:[{lbl:'sample', chips:[chip('blank', 'bad')]}],
    msg:{zh:'所以它什麼都生不出來。兩個 bug 的共同形狀是：**loss 變低了，而且是因為題目被改簡單了**。要抓它們只能看「模型最後要做的那件事」——這裡就是生成，再把生成結果丟回文法檢查。',
         en:'So it generates nothing. Both bugs have the same shape: the loss went down because the task got easier, not because the model got better. The only thing that catches them is evaluating what the model is ultimately for - here, generating, and then running the generated lines back through the grammar.'}});
  return F.list;
}

/* ================================================================ tab 5: attention */
const CODE_ATT = [
  'logits, cache = forward(p, cfg, ids)',
  'w = cache["layers"][1]["w"][0, 0]   # block 2, head 1',
  '',
  '# w[i, j] = how much position i read from position j',
  '# the upper triangle is empty: that is the causal mask'
];
function attStage(o){
  const sh = [], n = ATT.w.length, cw = 4.30 / n, X0 = 3.00, Y0 = 1.30;
  for (let i = 0; i < n; i++){
    for (let j = 0; j <= i; j++){
      const v = ATT.w[i][j];
      if (v > .03){
        sh.push(S.r(X0 + j * cw, Y0 + (n - 1 - i) * cw, cw * .9, cw * .9,
                    (o.row === i) ? 'hot' : 'ok', null));
      }
    }
  }
  if (o.row != null){
    sh.push(S.e(X0 - .12, Y0 + (n - 1 - o.row) * cw + cw / 2,
                X0 + n * cw + .12, Y0 + (n - 1 - o.row) * cw + cw / 2,
                {s:'ghost', arrow:false, w:.03}));
    sh.push(fitT(X0 - .22, Y0 + (n - 1 - o.row) * cw + cw / 2,
                 '"' + ATT.text[o.row] + '"', {anchor:'end', c:COL.hot, fs:.26}));
  }
  sh.push(fitT(X0 + n * cw / 2, Y0 - .34, ATT.text, {c:COL.grey, fs:.24}));
  return caps(sh, o);
}
function attFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:attStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({line:1,
    cap:{zh:'訓練完之後，一個 head 在看什麼', en:'what one head looks at once it is trained'},
    panels:[{lbl:'block / head', chips:[chip('2', 'dim'), chip('1', 'dim')]}],
    msg:{zh:'每一列是一個位置，亮的格子是它讀進來的位置。右上角整片空白就是 causal mask：位置 i 看不到 i 之後的任何東西。這張圖不是示意圖，是把 "' + ATT.text + '" 餵進訓練好的模型之後，從 cache 裡拿出來的那個權重矩陣。',
         en:'Each row is one position and the bright cells are where it read from. The empty upper triangle is the causal mask: position i cannot see anything after i. This is not a schematic - it is the weight matrix taken out of the cache after feeding "' + ATT.text + '" through the trained model.'}});
  [4, 12, 20].forEach(r => {
    p({row:r, line:3,
      cap:{zh:'第 ' + r + ' 個位置（字元 "' + ATT.text[r] + '"）往回看哪裡',
           en:'what position ' + r + ' (the character "' + ATT.text[r] + '") looks back at'},
      panels:[{lbl:'position', chips:[chip(String(r), 'hot')]},
              {lbl:'character', chips:[chip('"' + ATT.text[r] + '"', 'hot')]}],
      msg:{zh:'權重大多落在最近的幾個字元上，偶爾壓在比較前面的某一個——對一個只要「把詞拼完、然後接一個合法的下一段」的任務來說，這正是你會預期的樣子。不要過度解讀單一個 head：這裡只有兩層四個 head，它們是一起工作的。',
           en:'Most of the weight sits on the last few characters, with the occasional spike further back - which is what you would expect for a task that is mostly "finish this word, then start a legal next chunk". Do not over-read a single head: there are two layers of four here and they work together.'}});
  });
  return F.list;
}

/* ================================================================ meta */
const DAY_META = {
  title: {zh:'從零訓一個 char-level mini-GPT', en:'A character-level mini-GPT from scratch'},
  sub: {zh:SETUP.params.toLocaleString() + ' 個參數、numpy、手寫反向傳播，學會一套已知的文法',
        en:SETUP.params.toLocaleString() + ' parameters, numpy, a hand-written backward pass, and a grammar we know the entropy of'},
  tabs: [
    {id:'setup', label:{zh:'語料與模型', en:'the corpus and the model'},
     stage:{zh:'已知的產生器＝已知的底線', en:'a known generator means a known floor'},
     view:VIEW, code:CODE_SETUP, build:setupFrames,
     idea:{zh:'語料是五個空格的文法產生的 ' + SETUP.chars.toLocaleString() + ' 個字元、' + SETUP.vocab + ' 種字元。因為產生器已知，真正的熵也已知：' + SETUP.ent.toFixed(4) + ' bits/char，那是任何模型的下限。模型本身 ' + SETUP.params.toLocaleString() + ' 個參數，反向傳播手寫、跟中央差分差 ' + SETUP.grad.toExponential(0) + '。',
           en:'The corpus is ' + SETUP.chars.toLocaleString() + ' characters from a five-slot grammar over ' + SETUP.vocab + ' distinct characters. Because the generator is known, the true entropy is too: ' + SETUP.ent.toFixed(4) + ' bits per character, a floor no model can pass. The model has ' + SETUP.params.toLocaleString() + ' parameters and a hand-written backward pass that matches central differences to ' + SETUP.grad.toExponential(0) + '.'},
     legend:[['#ff9736', {zh:'目前這個空格', en:'the slot being filled'}],
             ['#12b3b8', {zh:'已經填過的', en:'slots already shown'}]]},
    {id:'train', label:{zh:'訓練', en:'training'},
     stage:{zh:'從 ln(vocab) 掉到接近熵', en:'from ln(vocab) down towards the entropy'},
     view:VIEW, code:CODE_TRAIN, build:trainFrames,
     idea:{zh:'500 步 Adam：loss 從 ' + TRAIN.hist[0].toFixed(4) + '（＝ln ' + SETUP.vocab + '，什麼都不知道）掉到 ' + TRAIN.final.toFixed(4) + '＝' + TRAIN.finalBits.toFixed(4) + ' bits/char，而底線是 ' + SETUP.ent.toFixed(4) + '。沒看過的同文法語料是 ' + TRAIN.valBits.toFixed(4) + ' bits/char，所以學到的是文法不是那份文字。',
           en:'500 Adam steps take the loss from ' + TRAIN.hist[0].toFixed(4) + ' - that is ln ' + SETUP.vocab + ', the cost of knowing nothing - down to ' + TRAIN.final.toFixed(4) + ' = ' + TRAIN.finalBits.toFixed(4) + ' bits per character, against a floor of ' + SETUP.ent.toFixed(4) + '. Unseen text from the same grammar scores ' + TRAIN.valBits.toFixed(4) + ', so what it learned is the grammar and not the text.'},
     legend:[['#12b3b8', {zh:'訓練 loss', en:'training loss'}],
             ['#ff9736', {zh:'產生器的熵', en:'the generator\'s entropy'}],
             ['#ff5c5c', {zh:'沒有 causal mask 的那條', en:'the run without the causal mask'}]]},
    {id:'sample', label:{zh:'生成', en:'sampling'},
     stage:{zh:'文法合格率，不是「看起來像」', en:'a grammar check, not a vibe'},
     view:VIEW, code:CODE_SAMPLE, build:sampleFrames,
     idea:{zh:'一次抽一個字元接回去。因為文法已知，每一行都可以被檢查「產生器產不產得出來」：temperature 1.0 / 0.8+top-k 5 / 0.2 分別是 ' + SAMPLES[0].frac.toFixed(0) + '% / ' + SAMPLES[1].frac.toFixed(0) + '% / ' + SAMPLES[2].frac.toFixed(0) + '%，而低溫買到的安全是用多樣性換的。',
           en:'Sample one character, append it, repeat. Because the grammar is known, every line can be tested against it: temperature 1.0, 0.8 with top-k 5, and 0.2 give ' + SAMPLES[0].frac.toFixed(0) + '%, ' + SAMPLES[1].frac.toFixed(0) + '% and ' + SAMPLES[2].frac.toFixed(0) + '% - and the safety a low temperature buys is paid for in variety.'},
     legend:[['#ff9736', {zh:'目前這個設定', en:'the setting being shown'}],
             ['#12b3b8', {zh:'合格的句子', en:'a grammatical line'}],
             ['#ff5c5c', {zh:'不合格的', en:'not grammatical'}]]},
    {id:'bugs', label:{zh:'兩個讓 loss 變低的 bug', en:'two bugs that lower the loss'},
     stage:{zh:'題目被改簡單了', en:'the task got easier, not the model better'},
     view:VIEW, code:CODE_BUG, build:bugFrames,
     idea:{zh:'忘了 causal mask：250 步 loss ' + TRAIN.noMaskLoss.toFixed(4) + ' 對 ' + TRAIN.maskedAt250.toFixed(4) + '，但生成合格率 ' + BAD.frac.toFixed(0) + '%。目標忘了移一格：loss ' + TRAIN.shiftLoss.toFixed(4) + '，模型學的是把輸入抄過去。兩個都讓 loss 變低、都不報錯，而且都只有在做「模型真正要做的事」時才看得出來。',
           en:'Forget the causal mask and 250 steps give ' + TRAIN.noMaskLoss.toFixed(4) + ' against ' + TRAIN.maskedAt250.toFixed(4) + ' - while the generated lines are ' + BAD.frac.toFixed(0) + '% grammatical. Forget to shift the targets and the loss is ' + TRAIN.shiftLoss.toFixed(4) + ' while the model learns to copy its input. Both lower the loss, neither raises anything, and only evaluating the real task finds them.'},
     legend:[['#12b3b8', {zh:'正確的訓練', en:'trained properly'}],
             ['#ff5c5c', {zh:'作弊版（loss 更低）', en:'the cheating run, with the lower loss'}]]},
    {id:'att', label:{zh:'它在看哪裡', en:'where it looks'},
     stage:{zh:'訓練完的注意力權重', en:'the attention weights after training'},
     view:VIEW, code:CODE_ATT, build:attFrames,
     idea:{zh:'把一句話餵進訓練好的模型，從 cache 拿出第二個 block、第一個 head 的權重矩陣。右上角整片空白就是 causal mask；權重大多壓在最近幾個字元，偶爾往回抓一個比較遠的。別從單一個 head 讀出太多故事——這裡只有兩層四個 head，它們是一起工作的。',
           en:'Feed one sentence through the trained model and take the weight matrix of block two, head one, out of the cache. The empty upper triangle is the causal mask; most of the weight sits on the last few characters, with occasional reaches further back. Do not read too much into a single head: there are two layers of four and they work together.'},
     legend:[['#12b3b8', {zh:'被讀到的位置', en:'a position that was read'}],
             ['#ff9736', {zh:'目前這一列', en:'the row being shown'}]]}
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
