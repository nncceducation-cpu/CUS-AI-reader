/* CUS Reader core — feature schema, label algebra, statistics, promotion
   gating and the fixed GMH-IVH paper rules.

   Nothing here touches the DOM, IndexedDB or TensorFlow.js, so every function
   in this file is exercised by tests/browser_learning.test.mjs under Node.
   Interval statistics were checked against statsmodels (Wilson) and
   scikit-learn (ROC AUC, Brier) before being written here. */

// ------------------------------------------------------------------ schema

/* Base findings that exist separately for each hemisphere. */
export const SIDED_FINDINGS = [
 ['gmh','germinal-matrix hemorrhage'],
 ['ivh','intraventricular blood'],
 ['distension','acute ventricular distension'],
 ['focal','focal periventricular echogenicity'],
 ['bright','PVE brighter than choroid'],
 ['inhomogeneous','inhomogeneous PVE'],
 ['cyst','periventricular cystic change']
];

/* Modelled heads. The first fifteen keys and their order are unchanged from
   0.7.0. The any_* heads are additions: a supplied category label such as
   "GMH-IVH Grade II" carries no hemisphere, and the 2026-09-30 audit found no
   hemisphere table for the 188 category-labelled studies, so without a
   side-agnostic target those studies can supply no training signal at all. */
export const FEATURES = [
 ['left_gmh','Left germinal-matrix hemorrhage'],['right_gmh','Right germinal-matrix hemorrhage'],
 ['left_ivh','Left intraventricular blood'],['right_ivh','Right intraventricular blood'],
 ['left_distension','Left acute ventricular distension'],['right_distension','Right acute ventricular distension'],
 ['left_focal','Left focal periventricular echogenicity'],['right_focal','Right focal periventricular echogenicity'],
 ['left_bright','Left PVE brighter than choroid'],['right_bright','Right PVE brighter than choroid'],
 ['left_inhomogeneous','Left inhomogeneous PVE'],['right_inhomogeneous','Right inhomogeneous PVE'],
 ['left_cyst','Left periventricular cystic change'],['right_cyst','Right periventricular cystic change'],
 ['cbh','Cerebellar hemorrhage'],
 ['any_gmh','Any-side germinal-matrix hemorrhage'],
 ['any_ivh','Any-side intraventricular blood'],
 ['any_distension','Any-side acute ventricular distension'],
 ['any_focal','Any-side focal periventricular echogenicity'],
 ['any_bright','Any-side PVE brighter than choroid'],
 ['any_inhomogeneous','Any-side inhomogeneous PVE'],
 ['any_cyst','Any-side periventricular cystic change']
];

export const FEATURE_KEYS = FEATURES.map(([key])=>key);
export const FEATURE_LABEL = Object.fromEntries(FEATURES);

/* 0.7.0 stored 64 px float frames. 0.8.0 stores 8-bit frames at a selectable
   edge and keeps the two generations separate: frames preprocessed by
   different pipelines must not be mixed inside one fit. */
export const EDGE = 64;
export const EDGES = [64, 96, 128];
export const DEFAULT_EDGE = 96;
export const SCHEMA = 'cus-browser-features-2';
export const LEGACY_SCHEMA = 'cus-browser-features-1';
export const PREP_LEGACY = 'v1-stretch-float64';
export const PREP_CURRENT = 'v2-letterbox-uint8';
export const VERSION = '0.8.0';

// --------------------------------------------------------- label algebra

/* Category supervision, unchanged from 0.7.0: it never invents laterality,
   negative findings in other domains, measurements or serial evidence. */
export function categoryAnnotations(category,laterality,normalComplete=false){
 if(category==='Normal'){
  if(!normalComplete)throw Error('Confirm a complete independently reviewed normal examination, including posterior fossa views, before marking all modeled findings absent.');
  return Object.fromEntries(FEATURES.map(([key])=>[key,0]));
 }
 if(category==='CBH')return {cbh:1};
 if(category.startsWith('GMH-IVH')){
  if(!['left','right','both'].includes(laterality))throw Error('Choose the expert-confirmed affected hemisphere for the supplied GMH-IVH label.');
  const sides=laterality==='both'?['left','right']:[laterality],labels={};
  for(const side of sides){
   if(category==='GMH-IVH Grade I'){labels[side+'_gmh']=1;labels[side+'_ivh']=0;}
   else{labels[side+'_ivh']=1;if(category==='GMH-IVH Grade III')labels[side+'_distension']=1;}
  }
  return labels;
 }
 throw Error('This category alone does not settle the modeled current-image findings. Record the verified findings below; serial diagnoses and measurements remain separate.');
}

/* Category supervision when the hemisphere is genuinely not recorded in the
   source dataset. Returns side-agnostic targets only. Left and right heads
   stay unknown, so laterality is never fabricated. */
export function categoryAnnotationsAnySide(category,normalComplete=false){
 if(category==='Normal')return categoryAnnotations('Normal','unknown',normalComplete);
 if(category==='CBH')return {cbh:1};
 if(category==='GMH-IVH Grade I')return {any_gmh:1,any_ivh:0};
 if(category==='GMH-IVH Grade II')return {any_ivh:1};
 if(category==='GMH-IVH Grade III')return {any_ivh:1,any_distension:1};
 throw Error('This category alone does not settle the modeled current-image findings. Record the verified findings below; serial diagnoses and measurements remain separate.');
}

/* any_X is determined by the hemispheres only when they settle it: either side
   positive makes it positive, both sides negative make it negative. One side
   negative and the other unknown leaves it unknown, and an explicit value
   supplied by the expert is preserved. */
export function deriveAnyLabels(labels){
 const out={...labels};
 for(const [base] of SIDED_FINDINGS){
  const left=labels['left_'+base],right=labels['right_'+base],key='any_'+base;
  if(left===1||right===1)out[key]=1;
  else if(left===0&&right===0)out[key]=0;
  else if(!(key in out))out[key]=null;
 }
 return out;
}

/* Reject annotations that contradict each other rather than silently picking
   one. Called by validateCase. */
export function labelConflicts(labels){
 const conflicts=[];
 for(const [base,name] of SIDED_FINDINGS){
  const left=labels['left_'+base],right=labels['right_'+base],any=labels['any_'+base];
  if(any===0&&(left===1||right===1))conflicts.push(`${name}: a hemisphere is marked present but the any-side head is marked absent.`);
  if(any===1&&left===0&&right===0)conflicts.push(`${name}: both hemispheres are marked absent but the any-side head is marked present.`);
 }
 if(labels.left_gmh===0&&labels.left_ivh===1)conflicts.push('Left intraventricular blood is marked present with germinal-matrix hemorrhage marked absent.');
 if(labels.right_gmh===0&&labels.right_ivh===1)conflicts.push('Right intraventricular blood is marked present with germinal-matrix hemorrhage marked absent.');
 return conflicts;
}

export function validateCase(record, existing=[]) {
 if(!record.infant?.trim() || !record.study?.trim()) throw Error('Enter a de-identified infant code and study code.');
 if(!['train','holdout','external'].includes(record.partition)) throw Error('Choose a dataset partition.');
 if(!record.blind) throw Error('Record the expert findings independently before viewing model output.');
 if(!record.modalityVerified || !record.axisVerified) throw Error('Confirm ultrasound content and frame-axis interpretation.');
 if(!record.frames?.length || !record.audit?.complete) throw Error('Complete frame decoding is required.');
 if(!Object.values(record.labels||{}).some(v=>v===0||v===1)) throw Error('Label at least one observable finding as present or absent. Unknown findings are excluded from training.');
 if(Object.entries(record.labels).some(([k,v])=>!FEATURES.some(([key])=>key===k)||![0,1,null].includes(v))) throw Error('Invalid feature annotations.');
 const conflicts=labelConflicts(record.labels);
 if(conflicts.length) throw Error('Contradictory findings. '+conflicts.join(' '));
 for(const old of existing) {
  if(old.id===record.id) continue;
  if(old.study===record.study) throw Error('This study code already exists. Remove the old record before replacing it.');
  if(old.infant===record.infant && old.partition!==record.partition) throw Error('All scans from one infant must stay in the same partition.');
  if(old.hashes.some(h=>record.hashes.includes(h))) throw Error('This source is already in the library; duplicate scans cannot be added as new cases.');
 }
}

export function eligibleFeatures(records) {
 return FEATURES.map(([key],i)=>({key,i,pos:new Set(records.filter(r=>r.labels[key]===1).map(r=>r.infant)).size,
  neg:new Set(records.filter(r=>r.labels[key]===0).map(r=>r.infant)).size})).filter(x=>x.pos>=2&&x.neg>=2);
}

// -------------------------------------------------- seeded pseudorandomness

/* 0.7.0 shuffled with sort(() => Math.random() - .5). A random comparator is
   not a uniform permutation; the outcome depends on the engine's sort
   internals. mulberry32 plus Fisher-Yates is uniform and reproducible, which
   also makes a trained version re-derivable from its recorded seed. */
export function mulberry32(seed){
 let a=seed>>>0;
 return function(){
  a=(a+0x6D2B79F5)>>>0;
  let t=Math.imul(a^(a>>>15),1|a)>>>0;
  t=((Math.imul(t^(t>>>7),61|t)>>>0)^t)>>>0;
  return ((t^(t>>>14))>>>0)/4294967296;
 };
}

export function shuffle(items,rand){
 const out=items.slice();
 for(let i=out.length-1;i>0;i--){
  const j=Math.floor(rand()*(i+1));
  const swap=out[i];out[i]=out[j];out[j]=swap;
 }
 return out;
}

// ------------------------------------------------------------- statistics

export const Z95 = 1.959963984540054;

/* Wilson score interval. Checked against
   statsmodels.stats.proportion.proportion_confint(method='wilson'). */
export function wilson(successes,total,z=Z95){
 if(!total)return null;
 const p=successes/total,d=1+z*z/total,c=p+z*z/(2*total),
  r=z*Math.sqrt(p*(1-p)/total+z*z/(4*total*total));
 return {lo:Math.max(0,(c-r)/d),hi:Math.min(1,(c+r)/d)};
}

function included(truth,scores){
 const y=[],s=[];
 truth.forEach((v,i)=>{if(v===0||v===1){y.push(v);s.push(scores[i]);}});
 return {y,s};
}

/* Rank-based ROC AUC with midranks for ties. Checked against
   sklearn.metrics.roc_auc_score, including all-tied and straddling-tie cases. */
export function auroc(truth,scores){
 const {y,s}=included(truth,scores);
 const nPos=y.filter(v=>v===1).length,nNeg=y.length-nPos;
 if(!nPos||!nNeg)return null;
 const order=s.map((_,i)=>i).sort((a,b)=>s[a]-s[b]);
 const ranks=new Array(s.length);
 for(let i=0;i<order.length;){
  let j=i;
  while(j+1<order.length&&s[order[j+1]]===s[order[i]])j++;
  const mid=(i+j)/2+1;
  for(let t=i;t<=j;t++)ranks[order[t]]=mid;
  i=j+1;
 }
 let rankSum=0;
 y.forEach((v,i)=>{if(v===1)rankSum+=ranks[i];});
 return (rankSum-nPos*(nPos+1)/2)/(nPos*nNeg);
}

/* Checked against sklearn.metrics.brier_score_loss. */
export function brier(truth,scores){
 const {y,s}=included(truth,scores);
 if(!y.length)return null;
 return y.reduce((total,v,i)=>total+(s[i]-v)**2,0)/y.length;
}

export function binaryMetrics(truth,scores,threshold=.5) {
 let tp=0,tn=0,fp=0,fn=0;
 truth.forEach((y,i)=>{const p=scores[i]>=threshold; if(y===1) p?tp++:fn++; else p?fp++:tn++;});
 const sensitivity=tp+fn?tp/(tp+fn):null, specificity=tn+fp?tn/(tn+fp):null;
 return {n:truth.length,positive:tp+fn,negative:tn+fp,tp,tn,fp,fn,sensitivity,specificity,
  balancedAccuracy:sensitivity!==null&&specificity!==null?(sensitivity+specificity)/2:null,
  threshold,
  sensitivityCI:tp+fn?wilson(tp,tp+fn):null,
  specificityCI:tn+fp?wilson(tn,tn+fp):null,
  auroc:auroc(truth,scores),brier:brier(truth,scores)};
}

/* Threshold maximising Youden's J. Intended for an inner tuning fold drawn
   from the training partition; using the development holdout here would
   contaminate the gate that the holdout exists to supply. */
export function youdenThreshold(truth,scores){
 const {y,s}=included(truth,scores);
 if(!y.some(v=>v===1)||!y.some(v=>v===0))return null;
 let best=null,bestJ=-Infinity;
 for(const t of [...new Set(s)].sort((a,b)=>a-b).concat([1.000001])){
  const m=binaryMetrics(y,s,t);
  if(m.sensitivity===null||m.specificity===null)continue;
  const j=m.sensitivity+m.specificity-1;
  if(j>bestJ+1e-12||(Math.abs(j-bestJ)<=1e-12&&best!==null&&Math.abs(t-.5)<Math.abs(best-.5))){
   best=t;bestJ=j;
  }
 }
 return best;
}

export function evaluation(records,predictions,trained,thresholds={}) {
 const domains={};
 for(const key of trained) {
  const y=[],p=[];records.forEach((r,i)=>{if(r.labels[key]===0||r.labels[key]===1){y.push(r.labels[key]);p.push(predictions[i][key]);}});
  domains[key]=binaryMetrics(y,p,thresholds[key]??.5);
 }
 const valid=Object.values(domains).filter(d=>d.balancedAccuracy!==null);
 const aucs=Object.values(domains).map(d=>d.auroc).filter(v=>v!==null);
 return {studies:records.length,infants:new Set(records.map(r=>r.infant)).size,domains,
  balancedAccuracy:valid.length?valid.reduce((s,d)=>s+d.balancedAccuracy,0)/valid.length:null,
  auroc:aucs.length?aucs.reduce((s,v)=>s+v,0)/aucs.length:null};
}

/* Bootstrap the difference in mean balanced accuracy between two models,
   resampling whole infants so serial scans from one infant are not treated as
   independent. This replaces 0.7.0's flat two-percentage-point comparison of
   two point estimates. */
export function clusterBootstrapDelta(records,predA,predB,trained,thresholds={},options={}){
 const {reps=2000,seed=20260930}=options;
 const groups=new Map();
 records.forEach((r,i)=>{
  if(!groups.has(r.infant))groups.set(r.infant,[]);
  groups.get(r.infant).push(i);
 });
 const keys=[...groups.keys()].sort();
 if(keys.length<2)return null;
 const rand=mulberry32(seed);
 const mean=(idx,pred)=>{
  const values=[];
  for(const key of trained){
   const y=[],p=[];
   for(const i of idx){
    if(records[i].labels[key]===0||records[i].labels[key]===1){y.push(records[i].labels[key]);p.push(pred[i][key]);}
   }
   const m=binaryMetrics(y,p,thresholds[key]??.5);
   if(m.balancedAccuracy!==null)values.push(m.balancedAccuracy);
  }
  return values.length?values.reduce((s,v)=>s+v,0)/values.length:null;
 };
 const all=records.map((_,i)=>i);
 const pointA=mean(all,predA),pointB=mean(all,predB);
 const deltas=[];
 for(let r=0;r<reps;r++){
  const idx=[];
  for(let k=0;k<keys.length;k++)idx.push(...groups.get(keys[Math.floor(rand()*keys.length)]));
  const a=mean(idx,predA),b=mean(idx,predB);
  if(a!==null&&b!==null)deltas.push(a-b);
 }
 if(deltas.length<Math.max(100,reps*.5))return null;
 deltas.sort((a,b)=>a-b);
 return {point:pointA!==null&&pointB!==null?pointA-pointB:null,
  lo:deltas[Math.floor(.025*deltas.length)],hi:deltas[Math.floor(.975*deltas.length)],
  reps:deltas.length,infants:keys.length};
}

/* Promotion gating.

   `allowed` keeps 0.7.0's meaning: the point estimates clear 80% sensitivity
   and specificity on an adequately populated holdout. `tier` is the honest
   reading of the same numbers. At the documented minimum of two positive and
   two negative studies, a perfect score has a Wilson lower bound of 0.34, and
   a coin-flip classifier clears the point-estimate gate 1 time in 16; a 0.80
   lower bound needs about 16 correct calls in each class. Requiring every
   learned finding to pass is an intersection of conditions and so is
   conservative, which is why no multiplicity adjustment is applied. */
export function promotionDecision(candidate,incumbent=null) {
 const domains=Object.values(candidate.domains);
 if(!domains.length || candidate.infants<4 || domains.some(d=>d.positive<2||d.negative<2)) return {allowed:false,tier:'insufficient',reason:'Need at least four held-out infants and two positive and two negative held-out studies for every learned finding.'};
 if(domains.some(d=>d.sensitivity<.8||d.specificity<.8)) return {allowed:false,tier:'insufficient',reason:'Every learned finding must reach 80% sensitivity and specificity on the development holdout.'};
 if(incumbent && (incumbent.balancedAccuracy===null || candidate.balancedAccuracy<incumbent.balancedAccuracy+.02)) return {allowed:false,tier:'insufficient',reason:'Candidate has not improved balanced accuracy by at least two percentage points over the active model on the same holdout.'};
 const bounded=domains.every(d=>d.sensitivityCI&&d.specificityCI);
 const qualified=bounded&&domains.every(d=>d.sensitivityCI.lo>=.8&&d.specificityCI.lo>=.8);
 if(qualified) return {allowed:true,tier:'development-qualified',
  reason:'Every learned finding holds a 95% Wilson lower bound at or above 80% sensitivity and specificity on the development holdout. Eligible for your research preview; external diagnostic validation remains outstanding.'};
 const worst=Math.min(...domains.map(d=>Math.min(d.sensitivityCI?.lo??0,d.specificityCI?.lo??0)));
 return {allowed:true,tier:'exploratory',
  reason:`Passed the point-estimate development gate, but the holdout is too small to establish performance: the weakest 95% Wilson lower bound across learned findings is ${(worst*100).toFixed(0)}%, against an 80% target. Usable for workflow testing only. Eligible for your research preview; external diagnostic validation remains outstanding.`};
}

export function studyFingerprint(records) {
 return records.map(r=>[r.study,r.infant,r.partition,[...(r.hashes||[])].sort().join(',')].join(':'))
  .sort().join('|');
}

// ------------------------------------------------------- fixed paper rules

/* Classify only expert-verified findings. Network scores never fill these
   fields. Grades and the strict 6 mm boundary are the paper's and are not
   changed by anything the model learns. */
export function expertGmhDetail(side) {
 if(side.hemorrhage==='unknown') return {grade:'Indeterminate: hemorrhage not settled',criterion:'Hemorrhage presence is required before any grade.',discordance:[]};
 if(side.hemorrhage==='no') return {grade:'No GMH-IVH',criterion:'No hemorrhage reported in this hemisphere.',discordance:[]};
 if(side.ivh==='no') return side.confined==='yes'
  ? {grade:'Grade I',criterion:'Hemorrhage confined to the germinal matrix with no intraventricular blood.',discordance:[]}
  : {grade:'Indeterminate: germinal-matrix confinement not confirmed',criterion:'Grade I requires confirmed germinal-matrix confinement.',discordance:[]};
 if(side.ivh!=='yes') return {grade:'Indeterminate: intraventricular blood not settled',criterion:'Intraventricular blood must be settled to separate Grade I from Grade II or III.',discordance:[]};
 const ahw=side.ahw===''?null:Number(side.ahw);
 if(ahw!==null && (!Number.isFinite(ahw)||ahw<0)) throw Error('AHW must be a non-negative verified measurement.');
 const discordance=[];
 if(side.distension==='yes'&&ahw!==null&&ahw<=6) discordance.push(`Acute distension was reported but the calibrated AHW is ${ahw} mm, which is not above the 6 mm Grade III threshold. Graded on the measurement; confirm the measurement and the distension assessment.`);
 if(side.distension==='no'&&ahw!==null&&ahw>6) discordance.push(`AHW is ${ahw} mm, above 6 mm, but acute distension was reported absent. Graded on the absent acute criterion; AHW above 6 mm without acute distension may reflect post-hemorrhagic change requiring serial assessment.`);
 if(side.distension==='no' || (ahw!==null&&ahw<=6)) return {grade:'Grade II',
  criterion:side.distension==='no'
   ? 'Intraventricular blood without acute ipsilateral ventricular distension.'
   : `Intraventricular blood with calibrated AHW ${ahw} mm, not above the 6 mm threshold.`,
  discordance};
 if(side.distension==='yes' && ahw!==null&&ahw>6) return {grade:'Grade III',
  criterion:`Intraventricular blood with acute ipsilateral distension and calibrated AHW ${ahw} mm, strictly above 6 mm.`,discordance};
 return {grade:'Indeterminate: acute distension and AHW are required',
  criterion:'Grade III requires both acute ipsilateral distension and a calibrated AHW strictly above 6 mm; Grade II requires an explicitly absent acute criterion.',
  discordance};
}

export function expertGmh(side) {
 return expertGmhDetail(side).grade;
}
