/* Executes the actual TensorFlow.js graph on the CPU backend.

   This file exists because shape reasoning is not enough. The 0.8.0 attention
   pooling initially normalised a [frames, 1] tensor along axis 0, which
   tf.softmax does not support; nothing short of running the graph catches
   that. It is kept separate from browser_learning.test.mjs so the dependency
   free suite still runs without the TensorFlow packages, and it skips rather
   than fails when they are absent.

   Install with: npm install    (see package.json devDependencies)
*/
import test from 'node:test';
import assert from 'node:assert/strict';

let tf, L, C, available = true, reason = '';
try {
  const core = await import('@tensorflow/tfjs-core');
  await import('@tensorflow/tfjs-backend-cpu');
  const layers = await import('@tensorflow/tfjs-layers');
  globalThis.tf = tf = Object.assign({}, core, layers);
  await core.setBackend('cpu');
  await core.ready();
  L = await import('../docs/assets/learning.js');
  C = await import('../docs/assets/core.js');
} catch (e) {
  available = false;
  reason = `TensorFlow.js packages not installed (${e.message.split('\n')[0]})`;
}

const EDGE = 64;
const EXPECTED_MAP = {64: 4, 96: 6, 128: 8};

function synthetic(code, positive, partition, rand) {
  return {
    id: code, infant: code, study: code, partition, plane: 'coronal', blind: true,
    modalityVerified: true, axisVerified: true, edge: EDGE, prep: C.PREP_CURRENT,
    category: 'synthetic', hashes: [code], audit: {complete: true},
    frames: Array.from({length: 4}, () => {
      const f = new Uint8Array(EDGE * EDGE);
      for (let i = 0; i < f.length; i++) f[i] = Math.floor(rand() * 60) + (positive && (i % EDGE) > EDGE / 2 ? 120 : 0);
      return f;
    }),
    labels: C.deriveAnyLabels({left_gmh: positive ? 1 : 0, right_gmh: positive ? 1 : 0})
  };
}
function cohort() {
  const rand = C.mulberry32(5), records = [];
  for (let i = 0; i < 4; i++) records.push(synthetic(`fit-pos-${i}`, true, 'train', rand),
                                           synthetic(`fit-neg-${i}`, false, 'train', rand));
  for (let i = 0; i < 2; i++) records.push(synthetic(`hold-pos-${i}`, true, 'holdout', rand),
                                           synthetic(`hold-neg-${i}`, false, 'holdout', rand));
  return records;
}

test('the encoder stack is valid at every supported resolution', {skip: !available && reason}, () => {
  for (const edge of C.EDGES) {
    const nets = L.createNetworks(edge, 1);
    const shape = nets.encoder.outputShape;
    assert.equal(shape[1], EXPECTED_MAP[edge], `spatial map at ${edge} px`);
    assert.equal(shape[2], EXPECTED_MAP[edge]);
    assert.equal(shape[3], L.SHAPES.CHANNELS);
    assert.equal(nets.attention.inputs[0].shape[1], L.SHAPES.EMBED);
    assert.equal(nets.head.inputs[0].shape[1], L.SHAPES.BAG);
    assert.equal(nets.head.outputs[0].shape[1], C.FEATURES.length);
    assert.ok(L.architectureMatches(nets, edge));
    assert.ok(!L.architectureMatches(nets, edge === 64 ? 96 : 64));
    L.disposeNetworks(nets);
  }
});

test('trainable parameters match the documented architecture', {skip: !available && reason}, () => {
  const nets = L.createNetworks(EDGE, 1);
  const total = [nets.encoder, nets.attention, nets.head].reduce((sum, model) =>
    sum + model.trainableWeights.reduce((t, w) => t + w.shape.reduce((a, b) => a * b, 1), 0), 0);
  assert.equal(total, 39879);
  L.disposeNetworks(nets);
});

test('a fit runs, scores are probabilities and attention normalises over frames',
  {skip: !available && reason}, async () => {
  const records = cohort();
  const before = tf.memory().numTensors;
  const version = await L.train(records, {epochs: 2, edge: EDGE, seed: 1, bagFrames: 0, augmentation: false}, () => {});
  try {
    assert.ok(version.trained.length > 0);
    assert.ok(version.trained.includes('any_gmh'), 'any-side heads must be learnable');
    assert.ok(version.history.every(h => Number.isFinite(h.loss)));
    assert.ok(version.training.tuneInfants > 0, 'an inner tuning fold should be used at eight infants');
    assert.match(version.thresholdSource, /inner tuning fold/);
    assert.ok(version.trained.every(k => Number.isFinite(version.thresholds[k])));
    assert.ok(Object.values(version.training.positiveWeights).every(w => w >= 0.25 && w <= 4));
    assert.ok(Object.values(version.metrics.domains).every(d => d.sensitivityCI && d.specificityCI));
    assert.ok(['insufficient', 'exploratory', 'development-qualified'].includes(version.gate.tier));

    const first = await L.predict(version.networks, records[0], version.trained, EDGE);
    assert.ok(Object.values(first.scores).every(v => Number.isFinite(v) && v >= 0 && v <= 1));
    assert.equal(first.attention.length, records[0].frames.length);
    assert.ok(Math.abs(first.attention.reduce((s, v) => s + v, 0) - 1) < 1e-5,
      'attention must be a distribution across frames');

    // dropout must be inactive at inference
    const again = await L.predict(version.networks, records[0], version.trained, EDGE);
    assert.deepEqual(first.scores, again.scores);

    // a longer bag still yields one weight per frame
    const long = {...records[0], frames: Array.from({length: 40}, () => new Uint8Array(EDGE * EDGE).fill(80))};
    assert.equal((await L.predict(version.networks, long, version.trained, EDGE)).attention.length, 40);

    // weights survive the persistence path unchanged
    const packed = await L.serialize(version);
    assert.ok(packed.encoder && packed.attention && packed.head);
    const reloaded = await L.restore({...packed, schema: C.SCHEMA});
    const after = await L.predict(reloaded.networks, records[0], version.trained, EDGE);
    for (const key of Object.keys(first.scores))
      assert.ok(Math.abs(first.scores[key] - after.scores[key]) < 1e-5, `${key} changed on reload`);
    L.disposeNetworks(reloaded.networks);
  } finally {
    L.disposeNetworks(version.networks);
  }
  assert.ok(tf.memory().numTensors - before < 200, 'tensors are leaking');
});

test('a recorded seed reproduces the same model', {skip: !available && reason}, async () => {
  const records = cohort();
  const options = {epochs: 2, edge: EDGE, seed: 7, bagFrames: 0, augmentation: false};
  const a = await L.train(records, options, () => {});
  const b = await L.train(records, options, () => {});
  const scoresA = await L.predict(a.networks, records[0], a.trained, EDGE);
  const scoresB = await L.predict(b.networks, records[0], b.trained, EDGE);
  for (const key of Object.keys(scoresA.scores))
    assert.ok(Math.abs(scoresA.scores[key] - scoresB.scores[key]) < 1e-6, `${key} not reproducible`);
  L.disposeNetworks(a.networks); L.disposeNetworks(b.networks);
});

test('augmentation, coronal mirroring and a capped bag all execute',
  {skip: !available && reason}, async () => {
  const version = await L.train(cohort(),
    {epochs: 2, edge: EDGE, seed: 3, bagFrames: 2, augmentation: true, coronalFlip: true}, () => {});
  assert.ok(version.history.every(h => Number.isFinite(h.loss)));
  assert.equal(version.training.bagFrames, 2);
  assert.equal(version.training.coronalFlip, true);
  L.disposeNetworks(version.networks);
});

test('mismatched resolution and too few infants are refused', {skip: !available && reason}, async () => {
  const records = cohort();
  await assert.rejects(
    () => L.train(records.map(r => ({...r, edge: 96})), {epochs: 1, edge: 64}, () => {}),
    /selected resolution/);
  // the first three records are three distinct infants, one short of the minimum
  await assert.rejects(() => L.train(records.slice(0, 3), {epochs: 1, edge: EDGE}, () => {}),
    /four independent infants/);
});
