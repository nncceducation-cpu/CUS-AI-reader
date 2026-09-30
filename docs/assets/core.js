export const FEATURES = [
 ['left_gmh','Left germinal-matrix hemorrhage'],['right_gmh','Right germinal-matrix hemorrhage'],
 ['left_ivh','Left intraventricular blood'],['right_ivh','Right intraventricular blood'],
 ['left_distension','Left acute ventricular distension'],['right_distension','Right acute ventricular distension'],
 ['left_focal','Left focal periventricular echogenicity'],['right_focal','Right focal periventricular echogenicity'],
 ['left_bright','Left PVE brighter than choroid'],['right_bright','Right PVE brighter than choroid'],
 ['left_inhomogeneous','Left inhomogeneous PVE'],['right_inhomogeneous','Right inhomogeneous PVE'],
 ['left_cyst','Left periventricular cystic change'],['right_cyst','Right periventricular cystic change'],
 ['cbh','Cerebellar hemorrhage']
];
export const EDGE=64, SCHEMA='cus-browser-features-1', VERSION='0.7.0';
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
export function validateCase(record, existing=[]) {
 if(!record.infant?.trim() || !record.study?.trim()) throw Error('Enter a de-identified infant code and study code.');
 if(!['train','holdout','external'].includes(record.partition)) throw Error('Choose a dataset partition.');
 if(!record.blind) throw Error('Record the expert findings independently before viewing model output.');
 if(!record.modalityVerified || !record.axisVerified) throw Error('Confirm ultrasound content and frame-axis interpretation.');
 if(!record.frames?.length || !record.audit?.complete) throw Error('Complete frame decoding is required.');
 if(!Object.values(record.labels||{}).some(v=>v===0||v===1)) throw Error('Label at least one observable finding as present or absent. Unknown findings are excluded from training.');
 if(Object.entries(record.labels).some(([k,v])=>!FEATURES.some(([key])=>key===k)||![0,1,null].includes(v))) throw Error('Invalid feature annotations.');
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
export function binaryMetrics(truth,scores,threshold=.5) {
 let tp=0,tn=0,fp=0,fn=0;
 truth.forEach((y,i)=>{const p=scores[i]>=threshold; if(y===1) p?tp++:fn++; else p?fp++:tn++;});
 const sensitivity=tp+fn?tp/(tp+fn):null, specificity=tn+fp?tn/(tn+fp):null;
 return {n:truth.length,positive:tp+fn,negative:tn+fp,tp,tn,fp,fn,sensitivity,specificity,
  balancedAccuracy:sensitivity!==null&&specificity!==null?(sensitivity+specificity)/2:null};
}
export function evaluation(records,predictions,trained) {
 const domains={};
 for(const key of trained) {
  const y=[],p=[];records.forEach((r,i)=>{if(r.labels[key]===0||r.labels[key]===1){y.push(r.labels[key]);p.push(predictions[i][key]);}});
  domains[key]=binaryMetrics(y,p);
 }
 const valid=Object.values(domains).filter(d=>d.balancedAccuracy!==null);
 return {studies:records.length,infants:new Set(records.map(r=>r.infant)).size,domains,
  balancedAccuracy:valid.length?valid.reduce((s,d)=>s+d.balancedAccuracy,0)/valid.length:null};
}
export function promotionDecision(candidate,incumbent=null) {
 const domains=Object.values(candidate.domains);
 if(!domains.length || candidate.infants<4 || domains.some(d=>d.positive<2||d.negative<2)) return {allowed:false,reason:'Need at least four held-out infants and two positive and two negative held-out studies for every learned finding.'};
 if(domains.some(d=>d.sensitivity<.8||d.specificity<.8)) return {allowed:false,reason:'Every learned finding must reach 80% sensitivity and specificity on the development holdout.'};
 if(incumbent && (incumbent.balancedAccuracy===null || candidate.balancedAccuracy<incumbent.balancedAccuracy+.02)) return {allowed:false,reason:'Candidate has not improved balanced accuracy by at least two percentage points over the active model on the same holdout.'};
 return {allowed:true,reason:'Passed the development gate. Eligible for your research preview; external diagnostic validation remains outstanding.'};
}
export function studyFingerprint(records) {return records.map(r=>r.id+':'+r.infant+':'+r.partition).sort().join('|');}
// Classify only expert-verified findings. Network scores never fill these fields.
export function expertGmh(side) {
 if(side.hemorrhage==='unknown') return 'Indeterminate: hemorrhage not settled';
 if(side.hemorrhage==='no') return 'No GMH-IVH';
 if(side.ivh==='no') return side.confined==='yes'?'Grade I':'Indeterminate: germinal-matrix confinement not confirmed';
 if(side.ivh!=='yes') return 'Indeterminate: intraventricular blood not settled';
 const ahw=side.ahw===''?null:Number(side.ahw);
 if(ahw!==null && (!Number.isFinite(ahw)||ahw<0)) throw Error('AHW must be a non-negative verified measurement.');
 if(side.distension==='no' || (ahw!==null&&ahw<=6)) return 'Grade II';
 if(side.distension==='yes' && ahw!==null&&ahw>6) return 'Grade III';
 return 'Indeterminate: acute distension and AHW are required';
}
