/* CUS Reader browser learning.

   A study is a bag of frames with one set of study-level labels, so this is
   multiple-instance learning. 0.7.0 pooled frames with a fixed mean and
   maximum over a 16-channel globally averaged map, which left a 32-number
   study representation and 2,543 trainable parameters: global average pooling
   over the whole frame is close to the worst possible summary for a focal
   lesion. 0.8.0 keeps spatial maximum pooling alongside the average, and
   learns which frames matter with gated attention pooling.

   None of this is a performance claim. It is a development baseline with
   enough capacity to be worth measuring. */

import {FEATURES,SCHEMA,EDGES,DEFAULT_EDGE,PREP_CURRENT,PREP_LEGACY,
        eligibleFeatures,evaluation,promotionDecision,studyFingerprint,
        mulberry32,shuffle,youdenThreshold} from './core.js';

const tf=globalThis.tf;

const FILTERS=[12,24,32,48];
const CHANNELS=FILTERS[FILTERS.length-1];
const EMBED=CHANNELS*2;        // spatial average and maximum per frame
const BAG=EMBED*2;             // attention-pooled and frame-wise maximum
const ATTENTION_UNITS=24;
const HEAD_UNITS=64;
const DROPOUT=.3;
const L2=1e-4;
const DEFAULT_BAG_FRAMES=64;

export async function ready(){
 if(!tf)throw Error('The learning library could not load. Refresh with an internet connection.');
 await tf.ready();
}
export function backend(){return tf?tf.getBackend():'unavailable';}

export function createNetworks(edge=DEFAULT_EDGE,seed=1){
 const init=offset=>tf.initializers.glorotUniform({seed:seed+offset});
 const reg=()=>tf.regularizers.l2({l2:L2});
 const encoder=tf.sequential();
 encoder.add(tf.layers.conv2d({inputShape:[edge,edge,1],filters:FILTERS[0],kernelSize:5,strides:2,
  padding:'same',activation:'relu',kernelInitializer:init(0),kernelRegularizer:reg()}));
 encoder.add(tf.layers.maxPooling2d({poolSize:2}));
 encoder.add(tf.layers.conv2d({filters:FILTERS[1],kernelSize:3,padding:'same',activation:'relu',
  kernelInitializer:init(1),kernelRegularizer:reg()}));
 encoder.add(tf.layers.maxPooling2d({poolSize:2}));
 encoder.add(tf.layers.conv2d({filters:FILTERS[2],kernelSize:3,padding:'same',activation:'relu',
  kernelInitializer:init(2),kernelRegularizer:reg()}));
 encoder.add(tf.layers.maxPooling2d({poolSize:2}));
 encoder.add(tf.layers.conv2d({filters:FILTERS[3],kernelSize:3,padding:'same',activation:'relu',
  kernelInitializer:init(3),kernelRegularizer:reg()}));

 /* Gated attention pooling over frames (Ilse et al. 2018), kept as a small
    dense network so the existing save and restore path still applies. */
 const attention=tf.sequential();
 attention.add(tf.layers.dense({inputShape:[EMBED],units:ATTENTION_UNITS,activation:'tanh',
  kernelInitializer:init(4)}));
 attention.add(tf.layers.dense({units:1,kernelInitializer:init(5)}));

 const head=tf.sequential();
 head.add(tf.layers.dense({inputShape:[BAG],units:HEAD_UNITS,activation:'relu',
  kernelInitializer:init(6),kernelRegularizer:reg()}));
 head.add(tf.layers.dropout({rate:DROPOUT,seed:seed+7}));
 head.add(tf.layers.dense({units:FEATURES.length,activation:'sigmoid',kernelInitializer:init(8)}));
 return {encoder,attention,head,edge};
}

export function disposeNetworks(networks){
 networks?.encoder?.dispose();networks?.attention?.dispose();networks?.head?.dispose();
}

function framesToTensor(frames,edge){
 const data=new Float32Array(frames.length*edge*edge);
 frames.forEach((frame,i)=>{
  const base=i*edge*edge,float=frame instanceof Float32Array;
  for(let p=0;p<edge*edge;p++)data[base+p]=float?frame[p]:frame[p]/255;
 });
 return tf.tensor4d(data,[frames.length,edge,edge,1]);
}

/* Returns the finding scores and the per-frame attention weight, so the
   reader can see which frames drove an estimate. */
/* The functional API is used throughout rather than chained tensor methods:
   chaining is registered by the full tfjs bundle but not by the modular core
   packages, so functional calls keep this module runnable under a plain Node
   test harness as well as in the browser. */
function forward(networks,frames,edge,training=false){
 const x=framesToTensor(frames,edge);
 const map=networks.encoder.apply(x,{training});
 const embedding=tf.concat([tf.mean(map,[1,2]),tf.max(map,[1,2])],1);   // [frames, EMBED]
 /* Attention must be normalised ACROSS frames. tf.softmax only supports the
    last dimension, so the [frames, 1] logits are transposed into [1, frames]
    for the softmax and back again. */
 const count=frames.length;
 const logits=networks.attention.apply(embedding,{training});           // [frames, 1]
 const weights=tf.reshape(tf.softmax(tf.reshape(logits,[1,count])),[count,1]);
 const pooled=tf.sum(tf.mul(embedding,weights),0);                      // [EMBED]
 const peak=tf.max(embedding,0);                                        // [EMBED]
 const scores=networks.head.apply(tf.reshape(tf.concat([pooled,peak]),[1,BAG]),{training});
 return {scores,weights};
}

export async function predict(networks,record,trained,edge){
 const side=edge||networks.edge||DEFAULT_EDGE;
 const {values,attention}=tf.tidy(()=>{
  const out=forward(networks,record.frames,side);
  return {values:out.scores.dataSync(),attention:out.weights.dataSync()};
 });
 if(Array.from(values).some(v=>!Number.isFinite(v)||v<0||v>1))
  throw Error('The model returned invalid scores; all estimates are withheld.');
 const scores={};
 FEATURES.forEach(([key],i)=>{if(trained.includes(key))scores[key]=values[i];});
 return {scores,attention:Array.from(attention)};
}

export async function measure(networks,records,trained,thresholds={},edge){
 const predictions=[];
 for(const record of records)predictions.push((await predict(networks,record,trained,edge)).scores);
 return {...evaluation(records,predictions,trained,thresholds),predictions};
}

function snapshot(networks){
 return ['encoder','attention','head'].map(name=>
  networks[name].getWeights().map(w=>({shape:w.shape,data:w.dataSync().slice()})));
}
function restoreSnapshot(networks,snap){
 ['encoder','attention','head'].forEach((name,i)=>{
  const tensors=snap[i].map(w=>tf.tensor(w.data,w.shape));
  networks[name].setWeights(tensors);
  tensors.forEach(t=>t.dispose());
 });
}

/* Horizontal flip is restricted to coronal studies. In a coronal view a
   mirror swaps the hemispheres, so flipping with the left and right labels
   exchanged is a valid extra example. In a parasagittal view the horizontal
   axis is anterior-posterior, so a flip would reverse anatomy without
   changing laterality at all. */
function flipLabels(labels){
 const out={...labels};
 for(const key of Object.keys(labels)){
  if(key.startsWith('left_'))out['right_'+key.slice(5)]=labels[key];
  else if(key.startsWith('right_'))out['left_'+key.slice(6)]=labels[key];
 }
 return out;
}
function augment(frames,edge,rand,options){
 const brightness=(rand()*2-1)*25;
 const contrast=1+(rand()*2-1)*.15;
 const dx=Math.round((rand()*2-1)*edge*.06),dy=Math.round((rand()*2-1)*edge*.06);
 const speckle=options.speckle?.04:0;
 const flip=options.flip;
 return frames.map(frame=>{
  const float=frame instanceof Float32Array,out=new Uint8Array(edge*edge);
  for(let y=0;y<edge;y++){
   const sy=y-dy;
   if(sy<0||sy>=edge)continue;
   for(let x=0;x<edge;x++){
    const sx0=x-dx;
    if(sx0<0||sx0>=edge)continue;
    const sx=flip?edge-1-sx0:sx0;
    let v=float?frame[sy*edge+sx]*255:frame[sy*edge+sx];
    v=(v-128)*contrast+128+brightness;
    if(speckle)v*=1+(rand()*2-1)*speckle;
    out[y*edge+x]=v<0?0:v>255?255:v;
   }
  }
  return out;
 });
}

function groupSplit(records,rand,fraction=.25){
 const infants=shuffle([...new Set(records.map(r=>r.infant))],rand);
 const take=Math.min(Math.max(1,Math.round(infants.length*fraction)),infants.length-4);
 if(take<1)return {fit:records,tune:[]};
 const held=new Set(infants.slice(0,take));
 return {fit:records.filter(r=>!held.has(r.infant)),tune:records.filter(r=>held.has(r.infant))};
}

export async function train(records,options,onProgress,signal){
 await ready();
 const {epochs=25,edge=DEFAULT_EDGE,seed=20260930,generation=PREP_CURRENT,
        bagFrames=DEFAULT_BAG_FRAMES,augmentation=true,coronalFlip=false}=options||{};
 if(!EDGES.includes(edge))throw Error('Unsupported learning resolution.');

 const pool=records.filter(r=>(r.prep||PREP_LEGACY)===generation);
 if(pool.some(r=>(r.edge||64)!==edge))
  throw Error('Every case in this fit must be stored at the selected resolution. Re-ingest the mismatched studies or change the resolution.');
 const development=pool.filter(r=>r.partition==='train'),holdout=pool.filter(r=>r.partition==='holdout');
 if(new Set(development.map(r=>r.infant)).size<4)
  throw Error('Add at least four independent infants to the training partition.');
 const eligible=eligibleFeatures(development),trained=eligible.map(e=>e.key);
 if(!trained.length)
  throw Error('At least one finding needs two positive and two negative training infants. Unknown labels are excluded.');

 /* Positive-class weights. Without these a finding present in a small
    minority of studies is best served by predicting absent for everything,
    which scores zero sensitivity and can never clear the gate. */
 const weightFor={};
 for(const e of eligible)weightFor[e.key]=Math.min(4,Math.max(.25,e.neg/Math.max(e.pos,1)));

 const rand=mulberry32(seed);
 /* Epoch and threshold selection use an inner split of the training
    partition, grouped by infant. The development holdout is never touched
    here: it exists to supply the gate, and selecting against it would
    invalidate the only independent measurement available. */
 const tuning=new Set(development.map(r=>r.infant)).size>=6?groupSplit(development,rand):{fit:development,tune:[]};
 const fitting=tuning.fit,tune=tuning.tune;

 const networks=createNetworks(edge,seed);
 const optimizer=tf.train.adam(.001);
 const variables=()=>[...networks.encoder.trainableWeights,...networks.attention.trainableWeights,
  ...networks.head.trainableWeights].map(w=>w.val);
 const history=[];
 let best=null,bestScore=-Infinity,bestEpoch=0;

 try{
  for(let epoch=0;epoch<epochs;epoch++){
   let loss=0,used=0;
   // One study is one optimisation step: a longer clip must not acquire extra
   // label weight just for having more frames.
   for(const record of shuffle(fitting,rand)){
    if(signal?.aborted)throw Error('Training stopped. The active model was unchanged.');
    let labels=record.labels;
    const mask=FEATURES.map(([k])=>trained.includes(k)&&[0,1].includes(labels[k])?1:0);
    if(!mask.some(Boolean))continue;

    let frames=record.frames;
    if(bagFrames&&frames.length>bagFrames)
     frames=shuffle(frames.map((f,i)=>i),rand).slice(0,bagFrames).sort((a,b)=>a-b).map(i=>record.frames[i]);
    const flip=coronalFlip&&record.plane==='coronal'&&rand()<.5;
    if(augmentation)frames=augment(frames,edge,rand,{speckle:true,flip});
    if(flip)labels=flipLabels(labels);

    const target=FEATURES.map(([k])=>labels[k]===1?1:0);
    const activeMask=FEATURES.map(([k])=>trained.includes(k)&&[0,1].includes(labels[k])?1:0);
    const posWeight=FEATURES.map(([k])=>weightFor[k]??1);

    const cost=optimizer.minimize(()=>tf.tidy(()=>{
     const p=tf.clipByValue(forward(networks,frames,edge,true).scores,1e-5,1-1e-5);
     const y=tf.tensor2d([target]),m=tf.tensor2d([activeMask]),w=tf.tensor2d([posWeight]);
     const one=tf.scalar(1);
     const positive=tf.mul(tf.mul(y,tf.log(p)),w);
     const negative=tf.mul(tf.sub(one,y),tf.log(tf.sub(one,p)));
     return tf.div(tf.neg(tf.sum(tf.mul(tf.add(positive,negative),m))),tf.sum(m));
    }),true,variables());
    const value=(await cost.data())[0];cost.dispose();
    if(!Number.isFinite(value))throw Error('Training became unstable. Candidate discarded; the active model was unchanged.');
    loss+=value;used++;
    await tf.nextFrame();
   }
   let tuneScore=null;
   if(tune.length){
    const m=await measure(networks,tune,trained,{},edge);
    tuneScore=m.balancedAccuracy;
    if(tuneScore!==null&&tuneScore>bestScore){
     bestScore=tuneScore;bestEpoch=epoch+1;best=snapshot(networks);
    }
   }
   history.push({epoch:epoch+1,loss:loss/Math.max(used,1),tuning:tuneScore});
   onProgress({epoch:epoch+1,epochs,loss:loss/Math.max(used,1),tuning:tuneScore,trained});
  }

  let selection='final epoch; no inner tuning fold was available';
  if(best){restoreSnapshot(networks,best);selection=`epoch ${bestEpoch} of ${epochs}, chosen on an inner tuning fold of ${new Set(tune.map(r=>r.infant)).size} training infants`;}

  /* Operating points come from the inner fold, never the holdout. */
  const thresholds={};let thresholdSource='fixed 0.50; no inner tuning fold was available';
  if(tune.length){
   const m=await measure(networks,tune,trained,{},edge);
   for(const key of trained){
    const y=[],p=[];
    tune.forEach((r,i)=>{if(r.labels[key]===0||r.labels[key]===1){y.push(r.labels[key]);p.push(m.predictions[i][key]);}});
    const t=youdenThreshold(y,p);
    thresholds[key]=t===null?.5:t;
   }
   thresholdSource=`Youden J on an inner tuning fold of ${new Set(tune.map(r=>r.infant)).size} training infants`;
  }else for(const key of trained)thresholds[key]=.5;

  const metrics=await measure(networks,holdout,trained,thresholds,edge);
  delete metrics.predictions;
  return {networks,trained,metrics,thresholds,thresholdSource,epochSelection:selection,history,
   edge,generation,seed,architecture:{filters:FILTERS,embed:EMBED,bag:BAG,
    attentionUnits:ATTENTION_UNITS,headUnits:HEAD_UNITS,dropout:DROPOUT,l2:L2,
    pooling:'spatial average and maximum per frame, gated attention and maximum across frames'},
   training:{epochs,bagFrames,augmentation,coronalFlip,backend:backend(),
    fitInfants:new Set(fitting.map(r=>r.infant)).size,
    tuneInfants:new Set(tune.map(r=>r.infant)).size,
    positiveWeights:weightFor},
   trainInfants:new Set(development.map(r=>r.infant)).size,
   trainStudies:development.length,datasetFingerprint:studyFingerprint(pool),
   trainedAt:new Date().toISOString(),schema:SCHEMA,
   trainingInfantCodes:[...new Set(development.map(r=>r.infant))],
   trainingSourceHashes:[...new Set(development.flatMap(r=>r.hashes))],
   status:'candidate',externalValidated:false,gate:promotionDecision(metrics)};
 }catch(error){disposeNetworks(networks);throw error;}
 finally{optimizer.dispose();}
}

async function artifacts(model){
 let saved;
 await model.save(tf.io.withSaveHandler(async a=>{
  saved=a;
  return {modelArtifactsInfo:{dateSaved:new Date(),modelTopologyType:'JSON',modelTopologyBytes:0,
   weightSpecsBytes:0,weightDataBytes:a.weightData.byteLength}};
 }));
 return saved;
}
export async function serialize(version){
 const {networks,...meta}=version;
 return {...meta,encoder:await artifacts(networks.encoder),
  attention:await artifacts(networks.attention),head:await artifacts(networks.head)};
}
export async function restore(version){
 await ready();
 if(version.schema!==SCHEMA)throw Error('Model schema is incompatible.');
 return {...version,networks:{
  encoder:await tf.loadLayersModel(tf.io.fromMemory(version.encoder)),
  attention:await tf.loadLayersModel(tf.io.fromMemory(version.attention)),
  head:await tf.loadLayersModel(tf.io.fromMemory(version.head)),
  edge:version.edge||DEFAULT_EDGE}};
}
export function architectureMatches(networks,edge){
 const input=networks.encoder.inputs[0].shape.slice(1).join(',');
 return input===`${edge},${edge},1`
  &&networks.attention.inputs[0].shape[1]===EMBED
  &&networks.head.inputs[0].shape[1]===BAG
  &&networks.head.outputs[0].shape[1]===FEATURES.length;
}
export const SHAPES={FILTERS,CHANNELS,EMBED,BAG,ATTENTION_UNITS,HEAD_UNITS,
 FEATURES:FEATURES.length,DEFAULT_BAG_FRAMES};
