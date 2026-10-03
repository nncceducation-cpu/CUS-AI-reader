import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {join, dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
import {validateCase,eligibleFeatures,binaryMetrics,evaluation,promotionDecision,expertGmh,categoryAnnotations,
        expertGmhDetail,categoryAnnotationsAnySide,deriveAnyLabels,labelConflicts,
        wilson,auroc,brier,youdenThreshold,mulberry32,shuffle,clusterBootstrapDelta,
        studyFingerprint,FEATURES} from '../docs/assets/core.js';
import {niftiData,windowBounds,letterbox,sectorBox,annotationScan} from '../docs/assets/media.js';
import {VECTORS,selfTests} from '../docs/assets/checks.js';
import {SHAPES} from '../docs/assets/learning.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');
const VECTOR_FILE = JSON.parse(readFileSync(join(HERE, 'fixtures', 'stat_vectors.json'), 'utf8'));
const near = (a, b, t = 1e-9) => a !== null && b !== null && Math.abs(a - b) <= t;

const record=(infant='A',partition='train')=>({id:infant,infant,study:infant,partition,blind:true,modalityVerified:true,axisVerified:true,frames:[new Float32Array(4096)],audit:{complete:true},hashes:[infant],labels:{left_gmh:1}});

// ---------------------------------------------------------------- 0.7.0 contract

test('category supervision does not invent laterality or absent findings',()=>{
 assert.throws(()=>categoryAnnotations('Normal','unknown'),/complete/);
 assert.throws(()=>categoryAnnotations('GMH-IVH Grade I','unknown'),/hemisphere/);
 assert.deepEqual(categoryAnnotations('GMH-IVH Grade I','left'),{left_gmh:1,left_ivh:0});
 assert.deepEqual(categoryAnnotations('GMH-IVH Grade II','right'),{right_ivh:1});
 assert.deepEqual(categoryAnnotations('GMH-IVH Grade III','left'),{left_ivh:1,left_distension:1});
 assert.throws(()=>categoryAnnotations('WMI','both'),/serial/);
});
test('infant leakage and duplicate sources are blocked',()=>{
 assert.throws(()=>validateCase({...record('A','holdout'),id:'different',study:'second'},[record()]),/same partition/);
 assert.throws(()=>validateCase({...record('B'),hashes:['A']},[record()]),/duplicate/i);
});
test('unknown findings cannot become negatives or unlock a head',()=>{
 assert.throws(()=>validateCase({...record(),labels:{left_gmh:null}}),/at least one/);
 const records=[record('A'),record('B'),{...record('C'),labels:{left_gmh:0}},{...record('D'),labels:{left_gmh:null}}];
 assert.equal(eligibleFeatures(records).length,0);
 records.push({...record('E'),labels:{left_gmh:0}});
 assert.deepEqual(eligibleFeatures(records).map(x=>x.key),['left_gmh']);
});
test('evaluation excludes unknown reference labels',()=>{
 const records=[record('A'),{...record('B'),labels:{left_gmh:0}},{...record('C'),labels:{left_gmh:null}}];
 const scores=[{left_gmh:.8},{left_gmh:.1},{left_gmh:.9}];
 const result=evaluation(records,scores,['left_gmh']);
 assert.equal(result.domains.left_gmh.n,2);assert.equal(result.balancedAccuracy,1);
});
test('promotion rejects absent classes, constant predictions and nonimprovement',()=>{
 assert.equal(promotionDecision({infants:4,domains:{x:binaryMetrics([0,0],[.1,.2])}}).allowed,false);
 const perfect=binaryMetrics([1,1,0,0],[.9,.8,.1,.2]);
 const result={infants:4,domains:{x:perfect},balancedAccuracy:1};
 assert.equal(promotionDecision(result).allowed,true);
 assert.equal(promotionDecision(result,{balancedAccuracy:1}).allowed,false);
 assert.equal(promotionDecision({...result,domains:{x:binaryMetrics([1,1,0,0],[.9,.9,.9,.9])}}).allowed,false);
});
test('GMH rules require verified AHW and keep strict six-mm boundary',()=>{
 const side={hemorrhage:'yes',ivh:'yes',distension:'yes',ahw:'6'};
 assert.equal(expertGmh(side),'Grade II');assert.equal(expertGmh({...side,ahw:'6.1'}),'Grade III');
 assert.match(expertGmh({...side,ahw:''}),/Indeterminate/);
 assert.equal(expertGmh({...side,distension:'no',ahw:''}),'Grade II');
 assert.throws(()=>expertGmh({...side,ahw:'-1'}),/non-negative/);
});
function fixture(frames=3){const buffer=new ArrayBuffer(352+8*7*frames*2),v=new DataView(buffer);v.setInt32(0,348,true);v.setInt16(40,3,true);v.setInt16(42,8,true);v.setInt16(44,7,true);v.setInt16(46,frames,true);v.setInt16(70,4,true);v.setFloat32(108,352,true);new Uint8Array(buffer,344,4).set([110,43,49,0]);for(let i=0;i<8*7*frames;i++)v.setInt16(352+i*2,Math.floor(i/56)*40,true);return buffer;}
test('NIfTI preserves source order and rejects truncated or over-budget input',()=>{
 const volume=niftiData(fixture());assert.equal(volume.n,3);assert.equal(volume.value(0),0);assert.equal(volume.value(56),40);assert.equal(volume.value(112),80);
 assert.throws(()=>niftiData(fixture().slice(0,370)),/Truncated/);
 assert.throws(()=>niftiData(fixture(257)),/no frames were sampled/);
});

// ------------------------------------------------------- 0.8.0 statistics

test('interval statistics match the reference implementations', () => {
 // Expectations generated by scripts/verify_numerics.py from statsmodels and
 // scikit-learn; the fixture is the recorded output of that script.
 for (const v of VECTOR_FILE.wilson) {
  const w = wilson(v.k, v.n);
  assert.ok(near(w.lo, v.lo), `Wilson lower ${v.k}/${v.n}: ${w.lo} vs ${v.lo}`);
  assert.ok(near(w.hi, Math.min(1, v.hi)), `Wilson upper ${v.k}/${v.n}: ${w.hi} vs ${v.hi}`);
 }
 for (const c of VECTOR_FILE.auroc) assert.ok(near(auroc(c.y, c.s), c.auc), `AUROC ${c.auc}`);
 assert.ok(near(brier(VECTOR_FILE.brier.y, VECTOR_FILE.brier.s), VECTOR_FILE.brier.value));
 assert.equal(youdenThreshold(VECTOR_FILE.youden.y, VECTOR_FILE.youden.s), VECTOR_FILE.youden.threshold);
 // null rather than a misleading number when a class is absent
 assert.equal(auroc([1, 1], [.9, .8]), null);
 assert.equal(wilson(0, 0), null);
});

test('the seeded shuffle is reproducible and uniform by construction', () => {
 const rand = mulberry32(VECTOR_FILE.prng.seed);
 const drawn = Array.from({length: 6}, () => rand());
 VECTOR_FILE.prng.first6.forEach((want, i) => assert.ok(near(drawn[i], want, 1e-12)));
 const permutation = shuffle([...Array(VECTOR_FILE.shuffle.n).keys()], mulberry32(VECTOR_FILE.shuffle.seed));
 assert.deepEqual(permutation, VECTOR_FILE.shuffle.result);
 // a permutation, not a resample
 assert.deepEqual([...permutation].sort((a, b) => a - b), [...Array(10).keys()]);
});

test('gating separates an exploratory signal from established performance', () => {
 const perfect = binaryMetrics([1, 1, 0, 0], [.9, .8, .1, .2]);
 const small = promotionDecision({infants: 4, domains: {x: perfect}, balancedAccuracy: 1});
 assert.equal(small.allowed, true);
 assert.equal(small.tier, 'exploratory');
 assert.match(small.reason, /too small/);
 // a perfect two-of-two score establishes only a 34% lower bound
 assert.ok(perfect.sensitivityCI.lo < .4);

 const n = VECTOR_FILE.nWilson80;
 assert.ok(wilson(n, n).lo >= .8, `${n} of ${n} should bound at 80%`);
 assert.ok(wilson(n - 1, n - 1).lo < .8, `${n - 1} of ${n - 1} should not`);

 const big = binaryMetrics(Array(20).fill(1).concat(Array(20).fill(0)),
                           Array(20).fill(.9).concat(Array(20).fill(.1)));
 const strong = promotionDecision({infants: 20, domains: {x: big}, balancedAccuracy: 1});
 assert.equal(strong.tier, 'development-qualified');
 // one weak finding downgrades the whole candidate: the gate is an intersection
 const mixed = promotionDecision({infants: 20, domains: {x: big, y: perfect}, balancedAccuracy: 1});
 assert.equal(mixed.tier, 'exploratory');
});

test('the replacement bootstrap clusters by infant', () => {
 const records = Array.from({length: 24}, (_, i) => ({infant: `I${Math.floor(i / 2)}`, labels: {x: i % 2}}));
 const strongPred = records.map(r => ({x: r.labels.x ? .9 : .1}));
 const weakPred = records.map((r, i) => ({x: (r.labels.x && i % 4 === 0) ? .9 : .45}));
 const better = clusterBootstrapDelta(records, strongPred, weakPred, ['x'], {}, {reps: 800, seed: 11});
 assert.ok(better.lo > 0, 'a real improvement should exclude zero');
 const same = clusterBootstrapDelta(records, strongPred, strongPred, ['x'], {}, {reps: 800, seed: 11});
 assert.ok(same.lo <= 0 && same.hi >= 0, 'a model against itself must contain zero');
 assert.equal(same.point, 0);
 // too few clusters to resample
 assert.equal(clusterBootstrapDelta([{infant: 'A', labels: {x: 1}}], [{x: .9}], [{x: .1}], ['x']), null);
});

test('a model fingerprint survives backup and restore', () => {
 const before = [{id: 'generated-1', study: 'S1', infant: 'A', partition: 'train', hashes: ['h1']}];
 const after = [{id: 'generated-2', study: 'S1', infant: 'A', partition: 'train', hashes: ['h1']}];
 assert.equal(studyFingerprint(before), studyFingerprint(after));
 const moved = [{id: 'generated-2', study: 'S1', infant: 'A', partition: 'holdout', hashes: ['h1']}];
 assert.notEqual(studyFingerprint(before), studyFingerprint(moved));
});

// --------------------------------------------------- 0.8.0 label algebra

test('category labels without a hemisphere supply any-side targets only', () => {
 assert.deepEqual(categoryAnnotationsAnySide('GMH-IVH Grade I'), {any_gmh: 1, any_ivh: 0});
 assert.deepEqual(categoryAnnotationsAnySide('GMH-IVH Grade II'), {any_ivh: 1});
 assert.deepEqual(categoryAnnotationsAnySide('GMH-IVH Grade III'), {any_ivh: 1, any_distension: 1});
 assert.deepEqual(categoryAnnotationsAnySide('CBH'), {cbh: 1});
 assert.throws(() => categoryAnnotationsAnySide('WMI'), /serial/);
 assert.throws(() => categoryAnnotationsAnySide('Normal'), /complete/);
 // no hemisphere key is ever produced
 for (const category of ['GMH-IVH Grade I', 'GMH-IVH Grade II', 'GMH-IVH Grade III'])
  assert.ok(!Object.keys(categoryAnnotationsAnySide(category)).some(k => /^(left|right)_/.test(k)));
});

test('any-side labels are derived only when the hemispheres settle them', () => {
 assert.equal(deriveAnyLabels({left_gmh: 1, right_gmh: 0}).any_gmh, 1);
 assert.equal(deriveAnyLabels({left_gmh: 0, right_gmh: 1}).any_gmh, 1);
 assert.equal(deriveAnyLabels({left_gmh: 0, right_gmh: 0}).any_gmh, 0);
 assert.equal(deriveAnyLabels({left_gmh: 0, right_gmh: null}).any_gmh, null);
 assert.equal(deriveAnyLabels({left_gmh: null, right_gmh: null}).any_gmh, null);
 // an expert-supplied any-side value is preserved when the sides are unknown
 assert.equal(deriveAnyLabels({any_ivh: 1}).any_ivh, 1);
 // deriving does not touch hemisphere labels
 assert.equal(deriveAnyLabels({any_ivh: 1}).left_ivh, undefined);
});

test('contradictory annotations are refused rather than silently resolved', () => {
 assert.equal(labelConflicts({left_gmh: 1, any_gmh: 0}).length, 1);
 assert.equal(labelConflicts({left_gmh: 0, right_gmh: 0, any_gmh: 1}).length, 1);
 assert.equal(labelConflicts({left_ivh: 1, left_gmh: 0}).length, 1);
 assert.equal(labelConflicts({left_gmh: 1, right_gmh: 0, any_gmh: 1}).length, 0);
 assert.throws(() => validateCase({...record(), labels: {left_gmh: 1, any_gmh: 0}}), /Contradictory/);
});

test('the modelled head list stays backward compatible', () => {
 const keys = FEATURES.map(([k]) => k);
 // the original fifteen keys keep their identity and position
 assert.deepEqual(keys.slice(0, 15), ['left_gmh','right_gmh','left_ivh','right_ivh','left_distension',
  'right_distension','left_focal','right_focal','left_bright','right_bright','left_inhomogeneous',
  'right_inhomogeneous','left_cyst','right_cyst','cbh']);
 assert.equal(keys.length, 22);
 assert.equal(new Set(keys).size, 22);
 assert.equal(SHAPES.FEATURES, 22);
});

// ------------------------------------------------- 0.8.0 clinical reporting

test('GMH-IVH output names its deciding criterion and flags discordance', () => {
 const side = {hemorrhage: 'yes', ivh: 'yes', distension: 'yes', ahw: '6'};
 const atBoundary = expertGmhDetail(side);
 assert.equal(atBoundary.grade, 'Grade II');
 assert.match(atBoundary.criterion, /6 mm/);
 assert.equal(atBoundary.discordance.length, 1, 'distension against a sub-threshold AHW must be flagged');
 const above = expertGmhDetail({...side, ahw: '6.1'});
 assert.equal(above.grade, 'Grade III');
 assert.equal(above.discordance.length, 0);
 // AHW above threshold with the acute criterion explicitly absent is also discordant
 const quiet = expertGmhDetail({...side, distension: 'no', ahw: '9'});
 assert.equal(quiet.grade, 'Grade II');
 assert.equal(quiet.discordance.length, 1);
 // every grade string still matches the 0.7.0 contract
 assert.equal(expertGmhDetail({hemorrhage: 'no'}).grade, 'No GMH-IVH');
 assert.match(expertGmhDetail({hemorrhage: 'unknown'}).grade, /Indeterminate/);
 assert.equal(expertGmhDetail({hemorrhage: 'yes', ivh: 'no', confined: 'yes'}).grade, 'Grade I');
 assert.match(expertGmhDetail({hemorrhage: 'yes', ivh: 'no', confined: 'unknown'}).grade, /confinement/);
});

// ------------------------------------------------------ 0.8.0 media pipeline

function nifti({w=8,h=7,d3=1,d4=1,dims=3,datatype=4,fill=null}={}) {
 const bytes = datatype === 16 ? 4 : 2;
 const n = d3 > 1 ? d3 : d4, count = w * h * Math.max(n, 1);
 const buffer = new ArrayBuffer(352 + count * bytes), v = new DataView(buffer);
 v.setInt32(0, 348, true);
 v.setInt16(40, dims, true); v.setInt16(42, w, true); v.setInt16(44, h, true);
 v.setInt16(46, d3, true); v.setInt16(48, d4, true);
 v.setInt16(70, datatype, true); v.setFloat32(108, 352, true);
 new Uint8Array(buffer, 344, 4).set([110, 43, 49, 0]);
 for (let i = 0; i < count; i++) {
  const value = fill ? fill(i, count) : Math.floor(i / (w * h)) * 40;
  if (datatype === 16) v.setFloat32(352 + i * 4, value, true); else v.setInt16(352 + i * 2, value, true);
 }
 return buffer;
}

test('a two-dimensional-plus-time NIfTI cine is decoded rather than rejected', () => {
 const cine = niftiData(nifti({dims: 4, d3: 1, d4: 5}));
 assert.equal(cine.n, 5);
 assert.match(cine.axis, /fourth \(time\) axis/);
 // a spatial stack declared with a singleton time axis still reads as spatial
 const spatial = niftiData(nifti({dims: 4, d3: 6, d4: 1}));
 assert.equal(spatial.n, 6);
 assert.match(spatial.axis, /third axis/);
 // two non-singleton axes remain genuinely ambiguous
 assert.throws(() => niftiData(nifti({dims: 4, d3: 3, d4: 2})), /Ambiguous/);
});

test('display windowing uses the data range instead of collapsing contrast', () => {
 // 0.7.0 decodeNifti passed any volume inside 0..255 through unscaled and
 // otherwise divided by Math.max(range, 1); both lose almost all contrast.
 const legacy = (x, min, max) => (min >= 0 && max <= 255) ? x : (x - min) / Math.max(max - min, 1) * 255;
 const legacySpan = (min, max) => legacy(max, min, max) - legacy(min, min, max);
 assert.equal(legacySpan(0, 12), 12, 'a 0 to 12 volume used 12 of 255 levels');
 assert.ok(Math.abs(legacySpan(0, 0.3) - 0.3) < 1e-12, 'a 0 to 0.3 float volume used 0.3 of 255 levels');

 const lowRange = Array.from({length: 4096}, (_, i) => i * 12 / 4095);
 const window = windowBounds(lowRange, 0, 12);
 assert.ok(window.hi - window.lo > 11, 'the window should span the data');
 assert.equal(window.degenerate, false);

 const sub = niftiData(nifti({dims: 3, d3: 3, datatype: 16, fill: (i, c) => i * 0.3 / (c - 1)}));
 assert.equal(sub.window.degenerate, false);
 assert.ok(sub.window.hi > sub.window.lo);
 // a constant volume degrades safely rather than dividing by zero
 assert.equal(niftiData(nifti({dims: 3, d3: 2, datatype: 16, fill: () => 5})).window.degenerate, true);
});

test('frames are letterboxed so source aspect ratio is preserved', () => {
 for (const c of VECTOR_FILE.letterbox) {
  const got = letterbox(c.w, c.h, 96);
  assert.deepEqual([got.w, got.h, got.x, got.y], c.out, `${c.w}x${c.h}`);
  assert.ok(Math.abs(got.w / got.h - c.w / c.h) < 0.02, 'aspect must be held');
  assert.ok(got.w <= 96 && got.h <= 96, 'must fit the square');
 }
 // the real supplied geometry: 733 x 494 was compressed by 1.48 in one axis
 const real = letterbox(733, 494, 96);
 assert.ok(Math.abs(real.w / real.h - 733 / 494) < 0.02);
});

test('sector cropping refuses to discard most of a frame', () => {
 const analysis = (aw, ah, fn) => {
  const grey = new Uint8Array(aw * ah);
  for (let y = 0; y < ah; y++) for (let x = 0; x < aw; x++) grey[y * aw + x] = fn(x, y);
  return {grey, aw, ah};
 };
 const middle = sectorBox(analysis(200, 200, (x, y) => (x > 50 && x < 150 && y > 50 && y < 150) ? 200 : 0), 400, 400);
 assert.ok(middle && middle.w > 150 && middle.w < 240, 'a central sector should be found');
 const full = sectorBox(analysis(200, 200, () => 200), 400, 400);
 assert.ok(full.fraction > 0.9, 'a fully lit frame must not be cropped');
 const speck = sectorBox(analysis(200, 200, (x, y) => (x > 98 && x < 102 && y > 98 && y < 102) ? 255 : 0), 400, 400);
 assert.equal(speck, null, 'a bright speck must not trigger an aggressive crop');
 const dark = sectorBox(analysis(200, 200, () => 0), 400, 400);
 assert.equal(dark, null);
 // the annotation screen is a bright-pixel heuristic, nothing more
 assert.equal(annotationScan(analysis(200, 200, (x, y) => (y < 12 && x % 3 === 0) ? 255 : 40)).suspected, true);
 assert.equal(annotationScan(analysis(200, 200, () => 120)).suspected, false);
});

// --------------------------------------------------- shipped artefact guards

test('the bundled self-tests all pass', () => {
 const results = selfTests();
 const failed = results.filter(r => !r.ok).map(r => r.name);
 assert.deepEqual(failed, [], 'in-browser self-tests must pass');
 assert.ok(results.length >= 20, `expected a meaningful panel, got ${results.length}`);
});

test('the self-test constants still match the generated vector file', () => {
 // Guards against someone editing checks.js expectations by hand so that the
 // panel agrees with a broken implementation.
 VECTORS.wilson.forEach((v, i) => {
  assert.equal(v.k, VECTOR_FILE.wilson[i].k);
  assert.ok(near(v.lo, VECTOR_FILE.wilson[i].lo, 1e-12));
  assert.ok(near(v.hi, Math.min(1, VECTOR_FILE.wilson[i].hi), 1e-12));
 });
 VECTORS.auroc.forEach((c, i) => assert.ok(near(c.auc, VECTOR_FILE.auroc[i].auc)));
 assert.equal(VECTORS.youden.threshold, VECTOR_FILE.youden.threshold);
 VECTORS.prng.first3.forEach((v, i) => assert.ok(near(v, VECTOR_FILE.prng.first6[i], 1e-12)));
 assert.deepEqual(VECTORS.shuffle.result, VECTOR_FILE.shuffle.result);
 assert.deepEqual(VECTORS.letterbox, VECTOR_FILE.letterbox);
 assert.equal(VECTORS.wilson80MinimumN, VECTOR_FILE.nWilson80);
});

test('the documented architecture matches the computed one', () => {
 assert.equal(SHAPES.EMBED, 2 * SHAPES.CHANNELS);
 assert.equal(SHAPES.BAG, 2 * SHAPES.EMBED);
 assert.equal(SHAPES.EMBED, 96);
 assert.equal(SHAPES.BAG, 192);
 // parameter counts recorded by scripts/verify_numerics.py
 const encoder = 5 * 5 * 1 * SHAPES.FILTERS[0] + SHAPES.FILTERS[0]
  + SHAPES.FILTERS.slice(1).reduce((total, f, i) => total + 3 * 3 * SHAPES.FILTERS[i] * f + f, 0);
 const attention = SHAPES.EMBED * SHAPES.ATTENTION_UNITS + SHAPES.ATTENTION_UNITS + SHAPES.ATTENTION_UNITS + 1;
 const head = SHAPES.BAG * SHAPES.HEAD_UNITS + SHAPES.HEAD_UNITS
  + SHAPES.HEAD_UNITS * SHAPES.FEATURES + SHAPES.FEATURES;
 assert.equal(encoder + attention + head, VECTOR_FILE.params.v080);
 assert.ok(VECTOR_FILE.params.v080 / VECTOR_FILE.params.v070 > 15);
});

test('the committed single-file edition is in sync with the modules', async () => {
 // The single-file build pins its own inline script hash in the CSP, so a
 // stale bundle is not merely out of date: it would block the page.
 const {buildSingleFile} = await import('../scripts/build_single_file.mjs');
 const built = buildSingleFile(ROOT);
 const committed = readFileSync(join(ROOT, 'docs', 'CUS-reader-single-file.html'), 'utf8');
 assert.equal(built.html, committed,
  'docs/CUS-reader-single-file.html is stale — run: node scripts/build_single_file.mjs .');
 // the CSP must pin the hash of the script actually embedded
 const inline = committed.match(/<script>([\s\S]*)<\/script>/)[1];
 const hash = `'sha256-${createHash('sha256').update(inline, 'utf8').digest('base64')}'`;
 assert.ok(committed.includes(hash), 'CSP script hash does not match the inline script');
 assert.ok(!/assets\//.test(committed), 'the bundle must not reference external assets');
});
