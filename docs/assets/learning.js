import {FEATURES,EDGE,SCHEMA,eligibleFeatures,evaluation,promotionDecision,studyFingerprint} from './core.js';
const tf=globalThis.tf;
export async function ready(){if(!tf)throw Error('The learning library could not load. Refresh with an internet connection.');await tf.ready();}
export function createNetworks(){
 const encoder=tf.sequential();
 encoder.add(tf.layers.conv2d({inputShape:[EDGE,EDGE,1],filters:8,kernelSize:5,strides:2,activation:'relu'}));
 encoder.add(tf.layers.maxPooling2d({poolSize:2}));
 encoder.add(tf.layers.conv2d({filters:16,kernelSize:3,activation:'relu'}));
 encoder.add(tf.layers.globalAveragePooling2d({dataFormat:'channelsLast'}));
 const head=tf.sequential();head.add(tf.layers.dense({inputShape:[32],units:24,activation:'relu'}));
 head.add(tf.layers.dense({units:FEATURES.length,activation:'sigmoid'}));
 return {encoder,head};
}
function forward(networks,frames,training=false){
 const data=new Float32Array(frames.length*EDGE*EDGE);frames.forEach((f,i)=>data.set(f,i*EDGE*EDGE));
 const x=tf.tensor4d(data,[frames.length,EDGE,EDGE,1]);
 const z=networks.encoder.apply(x,{training});
 const bag=tf.concat([z.mean(0),z.max(0)]).reshape([1,32]);
 return networks.head.apply(bag,{training});
}
export async function predict(networks,record,trained){
 const y=tf.tidy(()=>forward(networks,record.frames));
 const values=await y.data();y.dispose();
 if(Array.from(values).some(v=>!Number.isFinite(v)||v<0||v>1))throw Error('The model returned invalid scores; all estimates are withheld.');
 return Object.fromEntries(FEATURES.filter(([k])=>trained.includes(k)).map(([k],i)=>[k,values[FEATURES.findIndex(([key])=>key===k)]]));
}
export async function measure(networks,records,trained){const p=[];for(const record of records)p.push(await predict(networks,record,trained));return evaluation(records,p,trained);}
export async function train(records,epochs,onProgress,signal){
 await ready();
 const development=records.filter(r=>r.partition==='train'),holdout=records.filter(r=>r.partition==='holdout');
 if(new Set(development.map(r=>r.infant)).size<4)throw Error('Add at least four independent infants to the training partition.');
 const eligible=eligibleFeatures(development),trained=eligible.map(e=>e.key);
 if(!trained.length)throw Error('At least one finding needs two positive and two negative training infants. Unknown labels are excluded.');
 const networks=createNetworks(),optimizer=tf.train.adam(.001);
 try {
 for(let epoch=0;epoch<epochs;epoch++){
  let loss=0,used=0;
  // A whole study is one optimization step; long clips do not acquire extra label weight.
  for(const record of [...development].sort(()=>Math.random()-.5)){
   if(signal?.aborted)throw Error('Training stopped. The active model was unchanged.');
   const mask=FEATURES.map(([k])=>trained.includes(k)&&[0,1].includes(record.labels[k])?1:0);
   if(!mask.some(Boolean))continue;
   const target=FEATURES.map(([k])=>record.labels[k]===1?1:0);
   const cost=optimizer.minimize(()=>tf.tidy(()=>{
    const p=forward(networks,record.frames,true).clipByValue(.00001,.99999);
    const y=tf.tensor2d([target]),m=tf.tensor2d([mask]);
    return y.mul(p.log()).add(tf.scalar(1).sub(y).mul(tf.scalar(1).sub(p).log())).mul(m).sum().neg().div(m.sum());
   }),true,[...networks.encoder.trainableWeights,...networks.head.trainableWeights].map(w=>w.val));
   const value=(await cost.data())[0];cost.dispose();if(!Number.isFinite(value))throw Error('Training became unstable. Candidate discarded; the active model was unchanged.');loss+=value;used++;
   await tf.nextFrame();
  }
  onProgress({epoch:epoch+1,epochs,loss:loss/Math.max(used,1),trained});
 }
 const metrics=await measure(networks,holdout,trained);
 return {networks,trained,metrics,trainInfants:new Set(development.map(r=>r.infant)).size,
  trainStudies:development.length,datasetFingerprint:studyFingerprint(records),trainedAt:new Date().toISOString(),schema:SCHEMA,
  trainingInfantCodes:[...new Set(development.map(r=>r.infant))],trainingSourceHashes:[...new Set(development.flatMap(r=>r.hashes))],
  status:'candidate',externalValidated:false,gate:promotionDecision(metrics)};
 }catch(error){networks.encoder.dispose();networks.head.dispose();throw error;}finally{optimizer.dispose();}
}
async function artifacts(model){let saved;await model.save(tf.io.withSaveHandler(async a=>{saved=a;return {modelArtifactsInfo:{dateSaved:new Date(),modelTopologyType:'JSON',modelTopologyBytes:0,weightSpecsBytes:0,weightDataBytes:a.weightData.byteLength}};}));return saved;}
export async function serialize(version){const {networks,...meta}=version;return {...meta,encoder:await artifacts(networks.encoder),head:await artifacts(networks.head)};}
export async function restore(version){await ready();if(version.schema!==SCHEMA)throw Error('Model schema is incompatible.');return {...version,networks:{encoder:await tf.loadLayersModel(tf.io.fromMemory(version.encoder)),head:await tf.loadLayersModel(tf.io.fromMemory(version.head))}};}
