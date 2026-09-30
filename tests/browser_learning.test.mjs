import test from 'node:test';
import assert from 'node:assert/strict';
import {validateCase,eligibleFeatures,binaryMetrics,evaluation,promotionDecision,expertGmh,categoryAnnotations} from '../docs/assets/core.js';
import {niftiData} from '../docs/assets/media.js';
const record=(infant='A',partition='train')=>({id:infant,infant,study:infant,partition,blind:true,modalityVerified:true,axisVerified:true,frames:[new Float32Array(4096)],audit:{complete:true},hashes:[infant],labels:{left_gmh:1}});
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
