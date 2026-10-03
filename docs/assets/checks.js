/* Self-test vectors and checks, kept free of DOM and TensorFlow.js
   dependencies so the same assertions run in the browser panel and under the
   Node test suite.

   Expectations were generated from reference implementations:
   statsmodels.stats.proportion.proportion_confint(method='wilson') for the
   intervals, and sklearn.metrics.roc_auc_score / brier_score_loss for
   discrimination and calibration. tests/browser_learning.test.mjs asserts
   that these constants still match the generated vector file, so they cannot
   drift away from their source silently. */

import {wilson,auroc,brier,youdenThreshold,mulberry32,shuffle,binaryMetrics,
        promotionDecision,deriveAnyLabels,labelConflicts,expertGmhDetail} from './core.js';
import {letterbox} from './media.js';

export const VECTORS={
 wilson:[
  {k:4,n:4,lo:0.5101091635454027,hi:1},
  {k:8,n:10,lo:0.4901624715366418,hi:0.9433178485456247},
  {k:0,n:5,lo:0,hi:0.43448246478317476},
  {k:1,n:2,lo:0.09453120573423071,hi:0.9054687942657693},
  {k:16,n:20,lo:0.5839825677481065,hi:0.9193423374202021},
  {k:3,n:7,lo:0.15821985525146973,hi:0.7495416354723429}],
 auroc:[
  {y:[1,0,1,0,1],s:[.9,.1,.8,.4,.6],auc:1},
  {y:[1,1,0,0],s:[.5,.5,.5,.5],auc:.5},
  {y:[1,0,0,1,1,0],s:[.2,.2,.9,.9,.5,.5],auc:.5}],
 brier:{y:[1,0,1],s:[.9,.2,.6],value:.07},
 youden:{y:[1,1,0,0,1,0],s:[.8,.6,.55,.2,.9,.3],threshold:.6},
 prng:{seed:42,first3:[0.585752628511,0.592794201802,0.461197072873]},
 shuffle:{seed:7,n:10,result:[3,7,8,1,2,5,0,9,6,4]},
 letterbox:[{w:800,h:600,out:[96,72,0,12]},{w:1280,h:720,out:[96,54,0,21]},{w:512,h:512,out:[96,96,0,0]}],
 wilson80MinimumN:16};

export function selfTests(){
 const results=[];
 const near=(a,b,t=1e-9)=>a!==null&&b!==null&&Math.abs(a-b)<=t;
 const add=(name,ok,detail)=>results.push({name,ok,detail});

 for(const v of VECTORS.wilson){
  const w=wilson(v.k,v.n);
  add(`Wilson interval ${v.k}/${v.n}`,near(w.lo,v.lo)&&near(w.hi,Math.min(1,v.hi)),
   `${(w.lo*100).toFixed(2)}% to ${(w.hi*100).toFixed(2)}%`);
 }
 VECTORS.auroc.forEach((c,i)=>
  add(`ROC AUC case ${i+1}`,near(auroc(c.y,c.s),c.auc),String(auroc(c.y,c.s))));
 add('Brier score',near(brier(VECTORS.brier.y,VECTORS.brier.s),VECTORS.brier.value),
  String(brier(VECTORS.brier.y,VECTORS.brier.s)));
 add('Youden J threshold',youdenThreshold(VECTORS.youden.y,VECTORS.youden.s)===VECTORS.youden.threshold,
  String(youdenThreshold(VECTORS.youden.y,VECTORS.youden.s)));

 const rand=mulberry32(VECTORS.prng.seed),sequence=[rand(),rand(),rand()];
 add('Seeded PRNG stream',VECTORS.prng.first3.every((v,i)=>near(sequence[i],v,1e-11)),
  sequence.map(v=>v.toFixed(9)).join(', '));
 const permutation=shuffle([...Array(VECTORS.shuffle.n).keys()],mulberry32(VECTORS.shuffle.seed));
 add('Seeded Fisher-Yates shuffle',permutation.join(',')===VECTORS.shuffle.result.join(','),
  permutation.join(','));

 for(const c of VECTORS.letterbox){
  const got=letterbox(c.w,c.h,96);
  add(`Letterbox ${c.w}×${c.h} preserves aspect`,
   got.w===c.out[0]&&got.h===c.out[1]&&got.x===c.out[2]&&got.y===c.out[3],
   `${got.w}×${got.h} at (${got.x}, ${got.y})`);
 }

 const perfect=binaryMetrics([1,1,0,0],[.9,.8,.1,.2]);
 const smallGate=promotionDecision({infants:4,domains:{x:perfect},balancedAccuracy:1});
 add('A perfect 2-positive/2-negative holdout is exploratory only',smallGate.tier==='exploratory',
  `tier ${smallGate.tier}; sensitivity lower bound ${(perfect.sensitivityCI.lo*100).toFixed(0)}%`);
 const n=VECTORS.wilson80MinimumN;
 add(`${n} correct calls per class are needed for an 80% lower bound`,
  wilson(n,n).lo>=.8&&wilson(n-1,n-1).lo<.8,
  `${n} of ${n}: ${(wilson(n,n).lo*100).toFixed(1)}%; ${n-1} of ${n-1}: ${(wilson(n-1,n-1).lo*100).toFixed(1)}%`);

 add('Any-side labels derive only when the hemispheres settle them',
  deriveAnyLabels({left_gmh:1,right_gmh:0}).any_gmh===1
  &&deriveAnyLabels({left_gmh:0,right_gmh:0}).any_gmh===0
  &&deriveAnyLabels({left_gmh:0,right_gmh:null}).any_gmh===null,
  'one side positive gives present; both negative gives absent; one unknown stays unknown');
 add('Contradictory annotations are rejected',
  labelConflicts({left_gmh:1,any_gmh:0}).length===1,'one conflict detected');

 const boundary={hemorrhage:'yes',ivh:'yes',distension:'yes',ahw:'6'};
 add('The 6 mm Grade III boundary is strict',
  expertGmhDetail(boundary).grade==='Grade II'
  &&expertGmhDetail({...boundary,ahw:'6.1'}).grade==='Grade III',
  '6.0 mm is Grade II; 6.1 mm is Grade III');
 add('Acute distension against a sub-threshold AHW is flagged',
  expertGmhDetail(boundary).discordance.length===1,'discordance reported');
 return results;
}
