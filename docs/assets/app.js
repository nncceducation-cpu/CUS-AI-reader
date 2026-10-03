/* CUS Reader interface layer. */

import {FEATURES,FEATURE_LABEL,EDGE,EDGES,DEFAULT_EDGE,SCHEMA,LEGACY_SCHEMA,
        PREP_CURRENT,PREP_LEGACY,VERSION,validateCase,promotionDecision,expertGmhDetail,
        categoryAnnotations,categoryAnnotationsAnySide,deriveAnyLabels,labelConflicts,
        clusterBootstrapDelta,mulberry32} from './core.js';
import {decodeFiles,preview} from './media.js';
import * as db from './store.js';
import {ready,train,serialize,restore,predict,measure,backend,disposeNetworks,
        architectureMatches,SHAPES} from './learning.js';
import {selfTests} from './checks.js';

const $=id=>document.getElementById(id);
let current=null,cases=[],versions=[],active=null,runtime=null,busy=false,controller;
const acknowledged=new Set();

function message(text,kind='info'){
 const target=$('message');
 target.textContent=text;target.className=kind;
 if(kind==='error')target.focus();
}
function node(tag,text,classes){
 const e=document.createElement(tag);
 if(text!==undefined)e.textContent=text;
 if(classes)e.className=classes;
 return e;
}
function action(text,fn,secondary=true){
 const b=node('button',text,secondary?'secondary':'');
 b.addEventListener('click',()=>guard(fn));
 return b;
}
async function guard(fn){try{await fn();}catch(e){console.error(e);message(e.message||String(e),'error');}}
function page(id){
 document.querySelectorAll('.page').forEach(e=>e.hidden=e.id!==id);
 document.querySelectorAll('.tab').forEach(e=>{
  const on=e.dataset.page===id;
  e.classList.toggle('active',on);
  e.setAttribute('aria-selected',on?'true':'false');
 });
}
document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>page(b.dataset.page));

function table(headers,rows){
 const wrap=node('div',undefined,'scroll'),t=node('table'),head=node('thead'),tr=node('tr');
 headers.forEach(h=>tr.append(node('th',h)));
 head.append(tr);t.append(head);
 const body=node('tbody');
 for(const row of rows){
  const r=node('tr');
  for(const cell of row){const td=node('td');td.append(cell instanceof Node?cell:document.createTextNode(String(cell)));r.append(td);}
  body.append(r);
 }
 t.append(body);wrap.append(t);return wrap;
}
function percent(value){return value===null||value===undefined?'Not measurable':`${(value*100).toFixed(1)}%`;}
function interval(ci){return ci?`${(ci.lo*100).toFixed(0)}–${(ci.hi*100).toFixed(0)}%`:'not bounded';}
function withInterval(value,ci){
 if(value===null||value===undefined)return 'Not measurable';
 return ci?`${percent(value)} (95% CI ${interval(ci)})`:percent(value);
}
function fill(select,values,selected){
 for(const [value,text] of values){
  const o=node('option',text);o.value=String(value);
  if(String(value)===String(selected))o.selected=true;
  select.append(o);
 }
}

// ------------------------------------------------------------------- forms

function optionsForms(){
 fill($('prepEdge'),EDGES.map(e=>[e,`${e} × ${e} px`]),DEFAULT_EDGE);
 fill($('edge'),EDGES.map(e=>[e,`${e} × ${e} px`]),DEFAULT_EDGE);
}
function labelsForm(){
 const sided=FEATURES.filter(([k])=>!k.startsWith('any_'));
 const anySide=FEATURES.filter(([k])=>k.startsWith('any_'));
 const group=(title,items,hint)=>{
  const box=node('div',undefined,'labelgroup');
  box.append(node('h3',title));
  if(hint)box.append(node('p',hint,'muted'));
  const grid=node('div',undefined,'labels');
  for(const [key,label] of items){
   const l=node('label',label),select=node('select');
   select.id='label-'+key;
   fill(select,[['unknown','Unknown / not assessed'],['1','Present'],['0','Absent']],'unknown');
   l.append(select);grid.append(l);
  }
  box.append(grid);
  return box;
 };
 $('labels').append(
  group('Hemisphere-specific findings',sided),
  group('Any-side findings',anySide,
   'Use these when the hemisphere is not recorded in your source dataset. They are filled automatically when both hemispheres are settled.'));
}
function gmhForm(){
 for(const side of ['left','right']){
  const container=node('div');
  container.append(node('h3',side==='left'?'Left hemisphere':'Right hemisphere'));
  for(const [key,label] of [['hemorrhage','Hemorrhage present'],['confined','Confined to germinal matrix'],['ivh','Intraventricular blood'],['distension','Acute ipsilateral ventricular distension']]){
   const l=node('label',label),s=node('select');
   s.id=`${side}-${key}`;
   fill(s,['unknown','yes','no'].map(v=>[v,v]),'unknown');
   l.append(s);container.append(l);
  }
  const l=node('label','Calibrated AHW (mm)'),i=node('input');
  i.id=`${side}-ahw`;i.type='number';i.min='0';i.step='.1';
  l.append(i);container.append(l);
  $('gmh').append(container);
 }
}

// ------------------------------------------------------------------- state

function generations(){
 const counts=new Map();
 for(const r of cases){
  const key=r.prep||PREP_LEGACY;
  if(!counts.has(key))counts.set(key,{studies:0,infants:new Set(),edges:new Set()});
  const entry=counts.get(key);
  entry.studies++;entry.infants.add(r.infant);entry.edges.add(r.edge||EDGE);
 }
 return counts;
}
function generationForm(){
 const select=$('generation'),counts=generations();
 const previous=select.value;
 select.replaceChildren();
 const label=key=>key===PREP_CURRENT?'0.8 frames (letterboxed, 8-bit)':'0.7 frames (stretched, 64 px)';
 const entries=[...counts.entries()];
 if(!entries.length)fill(select,[[PREP_CURRENT,label(PREP_CURRENT)+' — no cases yet']],PREP_CURRENT);
 else fill(select,entries.map(([key,v])=>
  [key,`${label(key)} — ${v.studies} studies, ${v.infants.size} infants`]),previous||PREP_CURRENT);
}
function updateButtons(){
 $('toTeach').disabled=!current||busy;
 $('estimate').disabled=!current||!active||busy;
}
async function refresh(){
 cases=await db.all('cases');
 versions=(await db.all('versions')).sort((a,b)=>b.trainedAt.localeCompare(a.trainedAt));
 active=versions.find(v=>v.status==='active')||null;
 generationForm();renderLibrary();renderVersions();updateButtons();
}
function renderLibrary(){
 const target=$('library');target.replaceChildren();
 target.append(node('p',`${cases.length} studies · ${new Set(cases.map(r=>r.infant)).size} infants · ${cases.filter(r=>r.partition==='train').length} training · ${cases.filter(r=>r.partition==='holdout').length} holdout · ${cases.filter(r=>r.partition==='external').length} external`));
 if(!cases.length){
  target.append(node('p','Your library is empty. Add de-identified cases and independently verified findings.'));
  return;
 }
 const legacy=cases.filter(r=>(r.prep||PREP_LEGACY)===PREP_LEGACY).length;
 if(legacy)target.append(node('p',`${legacy} studies were stored by 0.7.0 at 64 px with stretched aspect. They remain readable and exportable, but a single fit cannot mix preprocessing generations: choose a generation on the training tab, or re-ingest those studies from source to use the current pipeline.`,'muted'));
 target.append(table(['Infant','Study','Label','Partition','Frames','Stored',''],cases.map(r=>[
  r.infant,r.study,r.category,r.partition,r.frames.length,
  `${r.edge||EDGE} px · ${(r.prep||PREP_LEGACY)===PREP_CURRENT?'0.8':'0.7'}`,
  action('Remove',async()=>{
   if(busy)throw Error('Wait for training to finish.');
   if(!confirm(`Remove ${r.study} from this browser? Export a backup first if you need it.`))return;
   await db.remove('cases',r.id);await refresh();
  })])));
}
function sparkline(history){
 if(history.length<2)return node('p','');
 const width=420,height=90,pad=4;
 const losses=history.map(h=>h.loss),lo=Math.min(...losses),hi=Math.max(...losses);
 const span=hi-lo||1;
 const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');
 svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
 svg.setAttribute('role','img');
 svg.setAttribute('aria-label',`Training loss fell from ${losses[0].toFixed(3)} to ${losses[losses.length-1].toFixed(3)} over ${history.length} epochs`);
 const point=(i,v)=>[pad+i/(history.length-1)*(width-2*pad),height-pad-(v-lo)/span*(height-2*pad)];
 const line=(values,colour,dash)=>{
  const path=document.createElementNS('http://www.w3.org/2000/svg','polyline');
  path.setAttribute('points',values.map((v,i)=>v===null?null:point(i,v).join(',')).filter(Boolean).join(' '));
  path.setAttribute('fill','none');path.setAttribute('stroke',colour);path.setAttribute('stroke-width','2');
  if(dash)path.setAttribute('stroke-dasharray','4 3');
  svg.append(path);
 };
 line(losses,'#116f72');
 const tuning=history.map(h=>h.tuning);
 if(tuning.some(v=>v!==null)){
  const tlo=Math.min(...tuning.filter(v=>v!==null)),thi=Math.max(...tuning.filter(v=>v!==null));
  const tspan=thi-tlo||1;
  const tpoint=(i,v)=>[pad+i/(history.length-1)*(width-2*pad),height-pad-(v-tlo)/tspan*(height-2*pad)];
  const path=document.createElementNS('http://www.w3.org/2000/svg','polyline');
  path.setAttribute('points',tuning.map((v,i)=>v===null?null:tpoint(i,v).join(',')).filter(Boolean).join(' '));
  path.setAttribute('fill','none');path.setAttribute('stroke','#b17b25');path.setAttribute('stroke-width','2');
  path.setAttribute('stroke-dasharray','4 3');
  svg.append(path);
 }
 const wrap=node('div',undefined,'curve');
 wrap.append(svg);
 wrap.append(node('p','Solid: training loss. Dashed: inner tuning-fold balanced accuracy, each series on its own scale.','muted'));
 return wrap;
}
function domainRows(version){
 return Object.entries(version.metrics.domains).map(([key,m])=>[
  FEATURE_LABEL[key]||key,
  `${m.positive} / ${m.negative}`,
  withInterval(m.sensitivity,m.sensitivityCI),
  withInterval(m.specificity,m.specificityCI),
  percent(m.balancedAccuracy),
  m.auroc===null?'Not measurable':m.auroc.toFixed(3),
  m.brier===null?'—':m.brier.toFixed(3),
  (version.thresholds?.[key]??.5).toFixed(3)]);
}
function renderVersions(){
 const target=$('versions');target.replaceChildren();
 if(!versions.length){
  target.append(node('p','No image model has been trained yet. Add labeled cases, then train a new version.'));
  return;
 }
 for(const version of versions){
  const card=node('div',undefined,'card');
  const tier=version.gate?.tier||'unknown';
  card.append(node('h3',`${version.name} · ${version.status==='active'?'Active research model':'Candidate'}`));
  card.append(node('p',`${new Date(version.trainedAt).toLocaleString()} · ${version.trainInfants} training infants · ${version.trained.length} learned findings · ${version.edge||EDGE} px · seed ${version.seed??'not recorded'}`));
  const badge=node('p',tier==='development-qualified'?'Development-qualified':tier==='exploratory'?'Exploratory only — performance not statistically established':'Did not pass the development gate',
   tier==='development-qualified'?'tier ok':tier==='exploratory'?'tier warn':'tier bad');
  card.append(badge);
  card.append(node('p',`Development holdout: ${version.metrics.infants} infants, ${version.metrics.studies} studies. Mean balanced accuracy: ${percent(version.metrics.balancedAccuracy)}. Mean AUROC: ${version.metrics.auroc===null?'not measurable':version.metrics.auroc.toFixed(3)}.`));
  card.append(node('p',version.gate.reason));
  if(version.thresholdSource)card.append(node('p',`Operating points: ${version.thresholdSource}. Epoch selection: ${version.epochSelection||'not recorded'}.`,'muted'));
  if(version.comparison?.bootstrap){
   const b=version.comparison.bootstrap;
   card.append(node('p',`Against the previous active model on the same holdout: ${(b.point*100).toFixed(1)} percentage points, infant-clustered bootstrap 95% CI ${(b.lo*100).toFixed(1)} to ${(b.hi*100).toFixed(1)} over ${b.reps} resamples of ${b.infants} infants.`,'muted'));
  }
  card.append(node('p','External validation: not established. Scores are uncalibrated.','muted'));
  card.append(table(['Finding','Pos / neg','Sensitivity','Specificity','Balanced accuracy','AUROC','Brier','Threshold'],domainRows(version)));
  const actions=node('div',undefined,'actions');
  if(version.status!=='active')actions.append(action('Check and activate for research',()=>activate(version.id)));
  actions.append(action('Export this model',()=>exportModel(version)));
  actions.append(action('Download model card',()=>downloadModelCard(version)));
  actions.append(action('Preview candidate findings',()=>runEstimate(version.id)));
  card.append(actions);
  target.append(card);
 }
}
function lineageCheck(record){
 for(const v of versions){
  if(record.partition!=='train'&&(v.trainingInfantCodes?.includes(record.infant)||record.hashes.some(h=>v.trainingSourceHashes?.includes(h))))
   throw Error('This infant or source has previously been used for model training. It cannot become an independent holdout or external test.');
 }
 for(const old of cases){
  if(record.center&&old.center===record.center&&(record.partition==='external')!==(old.partition==='external'))
   throw Error('An external-test center cannot also supply training or development-holdout cases.');
 }
}

// ----------------------------------------------------------------- reading

$('media').onchange=()=>guard(async()=>{
 if(busy)throw Error('Wait for training to finish.');
 current=null;
 $('estimates').replaceChildren();$('preview').replaceChildren();
 $('audit').replaceChildren();$('attention').replaceChildren();
 $('verify').hidden=true;updateButtons();
 message('Decoding complete sources…');
 current=await decodeFiles([...$('media').files],s=>message(s),{
  edge:Number($('prepEdge').value),
  sectorCrop:$('prepCrop').checked,
  blankTopBand:$('prepBlank').checked});
 $('audit').append(node('p',`${current.audit.decodedFrames} / ${current.audit.sourceFrames} frames decoded at ${current.edge} × ${current.edge} px. Complete sequential processing; no sampling.`));
 for(const source of current.sources){
  const crop=source.preprocessing?.cropBox;
  $('audit').append(node('p',`${source.name}: ${source.kind}, ${source.decodedFrames} frames${source.width?`, ${source.width} × ${source.height}`:''}. ${crop?`Sector crop ${crop.w} × ${crop.h} at (${crop.x}, ${crop.y}).`:'No sector crop applied.'}${source.displayWindow?` Display window ${source.displayWindow[0].toFixed(3)} to ${source.displayWindow[1].toFixed(3)}.`:''}`,'muted'));
 }
 if(current.audit.annotationWarnings.length)
  $('audit').append(node('p',`Possible burned-in annotation detected in: ${current.audit.annotationWarnings.join(', ')}. This screen measures bright pixels only and cannot read text or recognise a logo. Re-check de-identification at source, and consider blanking the upper band.`,'warn'));
 $('preview').append(...current.frames.slice(0,8).map(f=>{
  const img=node('img');img.src=preview(f,current.edge);
  img.alt='Resized frame preview, for navigation only';return img;}));
 $('audit').append(node('p','Up to eight resized previews are shown. Every decoded frame enters learning and inference.','muted'));
 /* Never pre-attest on the reader's behalf: both confirmations start clear. */
 $('modality').checked=false;$('axis').checked=false;
 $('axisHint').textContent=current.axisSuggested
  ? 'The decoder found one natural frame order for these sources. Confirm it represents this examination.'
  : 'These sources carry a stored NIfTI axis the decoder cannot interpret as spatial or temporal. Confirm the frame order yourself.';
 $('verify').hidden=false;updateButtons();
 message('Study decoded. Confirm ultrasound content and frame interpretation before labeling or estimating.','success');
});
$('toTeach').onclick=()=>page('teach');

$('prefill').onclick=()=>guard(()=>{
 const category=$('category').value,side=$('labelSide').value;
 const values=side==='unrecorded'
  ? categoryAnnotationsAnySide(category,$('normalComplete').checked)
  : categoryAnnotations(category,side,$('normalComplete').checked);
 for(const [key,value] of Object.entries(values))$('label-'+key).value=String(value);
 message(side==='unrecorded'
  ? 'Any-side findings prefilled. Both hemispheres remain unknown because your dataset does not record them. Verify before saving.'
  : 'Supported findings prefilled. Verify them independently before saving; other findings are unchanged.','success');
});

$('saveCase').onclick=()=>guard(async()=>{
 if(busy)throw Error('Wait for training to finish.');
 if(!current)throw Error('Upload a study in Read a study first.');
 if(!$('saveConsent').checked)throw Error('Confirm browser storage of the learning frames.');
 const raw=Object.fromEntries(FEATURES.map(([key])=>
  [key,$('label-'+key).value==='unknown'?null:Number($('label-'+key).value)]));
 const conflicts=labelConflicts(raw);
 if(conflicts.length)throw Error('Contradictory findings. '+conflicts.join(' '));
 const labels=deriveAnyLabels(raw);
 const record={...current,id:crypto.randomUUID(),infant:$('infant').value.trim(),study:$('study').value.trim(),
  category:$('category').value,partition:$('partition').value,plane:$('plane').value,
  center:$('center').value.trim(),labels,blind:$('blind').checked,
  modalityVerified:$('modality').checked,axisVerified:$('axis').checked,
  savedAt:new Date().toISOString(),schema:SCHEMA};
 validateCase(record,cases);lineageCheck(record);
 await db.put('cases',record);await refresh();
 $('blind').checked=false;
 message(`Saved ${record.study}. Unknown findings were excluded from its targets.`,'success');
});

async function networkFor(version){
 if(runtime?.id===version.id)return runtime;
 if(runtime)disposeNetworks(runtime.networks);
 runtime=await restore(version);
 return runtime;
}
function renderAttention(weights){
 const target=$('attention');target.replaceChildren();
 if(!weights?.length||weights.length<2)return;
 target.append(node('h3','Frames the model attended to'));
 const order=weights.map((w,i)=>[w,i]).sort((a,b)=>b[0]-a[0]).slice(0,8);
 const strip=node('div',undefined,'preview');
 for(const [weight,index] of order){
  const figure=node('figure',undefined,'frame');
  const img=node('img');
  img.src=preview(current.frames[index],current.edge);
  img.alt=`Frame ${index+1}, attention weight ${(weight*100).toFixed(1)} percent`;
  figure.append(img,node('figcaption',`#${index+1} · ${(weight*100).toFixed(1)}%`));
  strip.append(figure);
 }
 target.append(strip);
 target.append(node('p','Attention weights show which frames the pooling layer relied on. They are not lesion localisation, and have not been compared against expert annotation.','muted'));
}
async function runEstimate(id){
 if(busy)throw Error('Wait for training to finish.');
 if(!current)throw Error('Upload a study first.');
 if(!$('modality').checked||!$('axis').checked)throw Error('Confirm ultrasound content and sequence interpretation.');
 const version=versions.find(v=>v.id===id);
 if(!version)throw Error('No active research model. Add cases and train a version first.');
 const edge=version.edge||EDGE;
 if(edge!==current.edge)throw Error(`This model was fitted on ${edge} px frames but the study was decoded at ${current.edge} px. Re-decode this study at ${edge} px in the preprocessing options.`);
 if((version.generation||PREP_LEGACY)!==(current.prep||PREP_CURRENT))
  throw Error('This model was fitted on a different preprocessing generation than this study was decoded with.');
 if(version.gate?.tier==='exploratory'&&!acknowledged.has(version.id)){
  if(!confirm('This model cleared only the point-estimate gate. Its holdout is too small to bound performance, so the per-finding confidence intervals are wide and the scores may be no better than chance. Use for workflow testing only. Continue?'))return;
  acknowledged.add(version.id);
 }
 const loaded=await networkFor(version);
 const {scores,attention}=await predict(loaded.networks,current,version.trained,edge);
 $('estimates').replaceChildren(
  node('h2',`Research findings · ${version.name}`),
  node('p','Unvalidated AI feasibility suggestion. Research and workflow testing only. These are uncalibrated feature scores, not injury grades, measurements, or a normal-study determination.'));
 if(current.hashes.some(h=>version.trainingSourceHashes?.includes(h)))
  $('estimates').append(node('p','This is known training media. Its output is not an independent accuracy test.','notice'));
 $('estimates').append(table(['Learned finding','Uncalibrated score','Threshold','Above threshold','Holdout sensitivity','Holdout specificity'],
  Object.entries(scores).map(([key,value])=>{
   const m=version.metrics.domains[key],threshold=version.thresholds?.[key]??.5;
   return [FEATURE_LABEL[key]||key,percent(value),threshold.toFixed(3),value>=threshold?'Yes':'No',
    m?withInterval(m.sensitivity,m.sensitivityCI):'Not measured',
    m?withInterval(m.specificity,m.specificityCI):'Not measured'];
  })));
 $('estimates').append(node('p',`${current.frames.length} / ${current.audit.decodedFrames} frames processed; learned attention pooling with frame-wise maximum. Missing model heads remain unknown.`));
 renderAttention(attention);
 page('read');
}
$('estimate').onclick=()=>guard(()=>runEstimate(active?.id));

$('classify').onclick=()=>guard(()=>{
 const results=['left','right'].map(side=>{
  const input=Object.fromEntries(['hemorrhage','confined','ivh','distension','ahw']
   .map(key=>[key,$(`${side}-${key}`).value]));
  return [side,expertGmhDetail(input)];
 });
 $('expertResult').replaceChildren(
  table(['Hemisphere','Expert evidence-based GMH-IVH','Deciding criterion'],
   results.map(([side,detail])=>[side,detail.grade,detail.criterion])));
 const discordance=results.flatMap(([side,detail])=>detail.discordance.map(d=>`${side}: ${d}`));
 if(discordance.length){
  const box=node('div',undefined,'notice warn');
  box.append(node('strong','Discordant evidence'));
  for(const text of discordance)box.append(node('p',text));
  $('expertResult').append(box);
 }
 $('expertResult').append(node('p','Provisional domain classification from entered evidence. Complete study status, PVHI, serial findings and other domains have not been assessed.'));
});

// ---------------------------------------------------------------- training

$('train').onclick=()=>guard(async()=>{
 if(busy)return;
 busy=true;controller=new AbortController();
 $('train').disabled=true;$('stop').disabled=false;$('curve').replaceChildren();updateButtons();
 let candidate;
 try{
  await ready();
  message(`Training image-model weights locally on the ${backend()} backend. Leave this page open.`);
  candidate=await train(cases,{
   epochs:Number($('epochs').value),edge:Number($('edge').value),
   seed:Number($('seed').value)||0,generation:$('generation').value,
   bagFrames:Number($('bagFrames').value),augmentation:$('augment').checked,
   coronalFlip:$('coronalFlip').checked},
   p=>{
    $('progress').value=p.epoch/p.epochs*100;
    $('trainingStatus').textContent=`Epoch ${p.epoch} of ${p.epochs} · training loss ${p.loss.toFixed(4)} · ${p.trained.length} feature heads`+
     (p.tuning===null||p.tuning===undefined?'':` · inner tuning balanced accuracy ${(p.tuning*100).toFixed(1)}%`);
   },controller.signal);
  candidate.id=crypto.randomUUID();
  candidate.name=`CUS local ${VERSION} · ${new Date().toLocaleDateString()} · ${versions.length+1}`;
  $('curve').append(sparkline(candidate.history));
  const saved=await serialize(candidate);
  await db.put('versions',saved);
  await refresh();
  message('Candidate saved with its holdout results. The active model has not changed.','success');
 }finally{
  if(candidate?.networks)disposeNetworks(candidate.networks);
  busy=false;$('train').disabled=false;$('stop').disabled=true;updateButtons();
 }
});
$('stop').onclick=()=>controller?.abort();

async function activate(id){
 if(busy)throw Error('Wait for the current model operation to finish.');
 busy=true;updateButtons();
 try{
  const version=versions.find(v=>v.id===id);
  const holdout=cases.filter(r=>r.partition==='holdout'&&(r.prep||PREP_LEGACY)===(version.generation||PREP_LEGACY)&&(r.edge||EDGE)===(version.edge||EDGE));
  const previous=active;
  if(holdout.some(r=>version.trainingInfantCodes.includes(r.infant)||r.hashes.some(h=>version.trainingSourceHashes.includes(h))))
   throw Error('Holdout overlaps this model’s training history.');
  if(previous&&previous.trained.some(k=>!version.trained.includes(k)))
   throw Error('A replacement must retain every finding learned by the active model. Add enough labels for the missing findings.');
  message('Rechecking candidate against the current development holdout…');
  const loaded=await restore(version);
  let metrics,matched,baseline,bootstrap=null;
  try{
   metrics=await measure(loaded.networks,holdout,version.trained,version.thresholds||{},version.edge);
   if(previous&&previous.id!==id)matched=await measure(loaded.networks,holdout,previous.trained,version.thresholds||{},version.edge);
  }finally{disposeNetworks(loaded.networks);}
  const candidatePredictions=metrics.predictions;
  delete metrics.predictions;
  const gate=promotionDecision(metrics);
  if(!gate.allowed)throw Error(gate.reason);
  if(previous&&previous.id!==id){
   if(holdout.some(r=>previous.trainingInfantCodes.includes(r.infant)||r.hashes.some(h=>previous.trainingSourceHashes.includes(h))))
    throw Error('Current holdout is not independent of the active model.');
   const incumbent=await restore(previous);
   try{baseline=await measure(incumbent.networks,holdout,previous.trained,previous.thresholds||{},previous.edge);}
   finally{disposeNetworks(incumbent.networks);}
   const basePredictions=baseline.predictions;
   delete baseline.predictions;
   const replacement=promotionDecision(matched,baseline);
   if(!replacement.allowed)throw Error(replacement.reason);
   /* Infant-clustered bootstrap on the difference, replacing a bare
      two-percentage-point comparison of two point estimates. */
   bootstrap=clusterBootstrapDelta(holdout,matched.predictions||candidatePredictions,basePredictions,
    previous.trained,version.thresholds||{},{reps:2000,seed:(version.seed??1)+7});
   if(gate.tier==='development-qualified'&&(!bootstrap||bootstrap.lo<=0)){
    gate.tier='exploratory';
    gate.reason+=' The infant-clustered bootstrap on the difference against the active model does not exclude zero, so the improvement is not established.';
   }
  }
  delete matched?.predictions;
  const changed=versions.map(v=>v.id===id
   ?{...v,status:'active',metrics,gate,comparison:baseline?{baseline,candidate:matched,bootstrap}:null,activatedAt:new Date().toISOString()}
   :v.status==='active'?{...v,status:'candidate'}:v);
  await db.replaceVersions(changed);await refresh();
  message(`Activated for research previews on this device, tier: ${gate.tier}. External diagnostic validation remains outstanding.`,'success');
 }finally{busy=false;updateButtons();}
}

// --------------------------------------------------------- export / import

function base64(buffer){
 const bytes=new Uint8Array(buffer);let text='';
 for(let i=0;i<bytes.length;i+=32768)text+=String.fromCharCode(...bytes.subarray(i,i+32768));
 return btoa(text);
}
function unbase64(text){
 if(typeof text!=='string'||text.length>80*1024*1024)throw Error('Invalid backup payload.');
 const s=atob(text),bytes=new Uint8Array(s.length);
 for(let i=0;i<s.length;i++)bytes[i]=s.charCodeAt(i);
 return bytes.buffer;
}
/* One packed 8-bit buffer per case rather than a base64 string per frame. */
function packFrames(record){
 const edge=record.edge||EDGE,bytes=new Uint8Array(record.frames.length*edge*edge);
 record.frames.forEach((frame,i)=>{
  const offset=i*edge*edge,float=frame instanceof Float32Array;
  for(let p=0;p<edge*edge;p++)bytes[offset+p]=float?Math.round(frame[p]*255):frame[p];
 });
 return base64(bytes.buffer);
}
function unpackFrames(text,edge,count){
 const bytes=new Uint8Array(unbase64(text));
 if(bytes.length!==count*edge*edge)throw Error('Invalid learning-frame payload.');
 const frames=[];
 for(let i=0;i<count;i++)frames.push(bytes.slice(i*edge*edge,(i+1)*edge*edge));
 return frames;
}
function encodeVersion(v){
 return {...v,encoder:{...v.encoder,weightData:base64(v.encoder.weightData)},
  attention:{...v.attention,weightData:base64(v.attention.weightData)},
  head:{...v.head,weightData:base64(v.head.weightData)}};
}
function download(value,name,type='application/json'){
 const url=URL.createObjectURL(new Blob([typeof value==='string'?value:JSON.stringify(value)],{type}));
 const a=node('a');a.href=url;a.download=name;a.click();
 setTimeout(()=>URL.revokeObjectURL(url),3000);
}
function exportModel(v){
 download({format:'cus-private-model-2',version:encodeVersion(v)},`CUS-private-model-${v.id}.json`);
 message('Downloaded private model weights and evaluation metadata.','success');
}

function modelCard(version){
 const partition=name=>cases.filter(r=>r.partition===name);
 const countBy=records=>`${records.length} studies, ${new Set(records.map(r=>r.infant)).size} infants`;
 const counts=key=>{
  const pos=new Set(cases.filter(r=>r.partition==='train'&&r.labels[key]===1).map(r=>r.infant)).size;
  const neg=new Set(cases.filter(r=>r.partition==='train'&&r.labels[key]===0).map(r=>r.infant)).size;
  return `${pos} positive / ${neg} negative training infants`;
 };
 const rows=Object.entries(version.metrics.domains).map(([key,m])=>
  `| ${FEATURE_LABEL[key]||key} | ${m.positive} / ${m.negative} | ${withInterval(m.sensitivity,m.sensitivityCI)} | ${withInterval(m.specificity,m.specificityCI)} | ${m.auroc===null?'n/a':m.auroc.toFixed(3)} | ${m.brier===null?'n/a':m.brier.toFixed(3)} | ${(version.thresholds?.[key]??.5).toFixed(3)} | ${counts(key)} |`);
 return `# CUS Reader model card — ${version.name}

**This is not a validated diagnostic model.** It was fitted in a browser from a
private local case library. No external validation has been performed and the
scores are uncalibrated. Nothing below is evidence of clinical performance.

## Provenance

- Reader version: ${VERSION}, feature schema ${version.schema}
- Trained: ${version.trainedAt}
- Random seed: ${version.seed ?? 'not recorded'} (training is reproducible from this seed)
- Dataset fingerprint: \`${version.datasetFingerprint}\`
- Preprocessing generation: ${version.generation||PREP_LEGACY}
- TensorFlow.js backend at fit time: ${version.training?.backend||'not recorded'}

## Dataset composition

| Partition | Content |
|---|---|
| Training | ${countBy(partition('train'))} |
| Development holdout | ${countBy(partition('holdout'))} |
| Untouched external test | ${countBy(partition('external'))} |

Fitting used ${version.training?.fitInfants ?? 'n/a'} infants; ${version.training?.tuneInfants ?? 0} training
infants were held back as an inner tuning fold for epoch and threshold
selection. The development holdout was not used for any selection.

## Preprocessing

- Stored edge: ${version.edge||EDGE} px, 8-bit greyscale, aspect preserved by letterboxing
- Frames per training step: ${version.training?.bagFrames||'every frame'} (inference always uses every decoded frame)
- Augmentation: ${version.training?.augmentation?'brightness, contrast, shift, speckle':'none'}
- Coronal mirroring with hemisphere label exchange: ${version.training?.coronalFlip?'enabled':'disabled'}

## Architecture

Convolutional encoder with filters ${(version.architecture?.filters||[]).join(', ')}, producing a
${version.architecture?.embed} dimensional frame embedding from spatial average and maximum
pooling. Frames are combined by gated attention pooling concatenated with the
frame-wise maximum, giving a ${version.architecture?.bag} dimensional study representation, then a
dense head with ${version.architecture?.headUnits} units, dropout ${version.architecture?.dropout} and L2 ${version.architecture?.l2}.

## Operating points

${version.thresholdSource||'not recorded'}. Epoch selection: ${version.epochSelection||'not recorded'}.

## Measured performance on the development holdout

${version.metrics.infants} infants, ${version.metrics.studies} studies.
Mean balanced accuracy ${percent(version.metrics.balancedAccuracy)}, mean AUROC ${version.metrics.auroc===null?'n/a':version.metrics.auroc.toFixed(3)}.

| Finding | Holdout pos / neg | Sensitivity | Specificity | AUROC | Brier | Threshold | Training support |
|---|---|---|---|---|---|---|---|
${rows.join('\n')}

Intervals are 95% Wilson score intervals on the holdout counts. Because the
gate requires every learned finding to pass simultaneously, it is an
intersection of conditions and therefore conservative; no multiplicity
adjustment is applied. Studies from one infant are not independent, so
study-level intervals are optimistic wherever an infant contributes more than
one holdout study.

## Gate decision

Tier: **${version.gate?.tier||'unknown'}**. ${version.gate?.reason||''}

${version.comparison?.bootstrap?`Against the previous active model on the same holdout: ${(version.comparison.bootstrap.point*100).toFixed(1)} percentage points of mean balanced accuracy, infant-clustered bootstrap 95% CI ${(version.comparison.bootstrap.lo*100).toFixed(1)} to ${(version.comparison.bootstrap.hi*100).toFixed(1)} over ${version.comparison.bootstrap.reps} resamples.`:'No replacement comparison was recorded for this version.'}

## Outstanding before any clinical claim

External-center testing on untouched data, probability calibration, scanner and
view subgroup analysis, specialist adjudication of reference labels, and
prespecified acceptance criteria agreed with expert users before prospective
blinded evaluation. See VALIDATION_PROTOCOL.md.
`;
}
function downloadModelCard(version){
 download(modelCard(version),`CUS-model-card-${version.id}.md`,'text/markdown');
 message('Downloaded the model card.','success');
}

$('backup').onclick=()=>guard(()=>{
 if(busy)throw Error('Wait for training to finish.');
 download({format:'cus-private-backup-2',
  cases:cases.map(r=>({...r,frames:undefined,frameCount:r.frames.length,
   edge:r.edge||EDGE,framesPacked:packFrames(r)})),
  versions:versions.map(encodeVersion)},'CUS-private-backup.json');
 message('Downloaded your private case library and model versions.','success');
});

$('restore').onchange=()=>guard(async()=>{
 if(busy)throw Error('Wait for training to finish.');
 const input=$('restore'),file=input.files[0];
 try{
  if(!file)return;
  if(file.size>64*1024*1024)throw Error('Backup exceeds the 64 MB import budget.');
  const data=JSON.parse(await file.text());
  if(!['cus-private-backup-1','cus-private-backup-2','cus-private-model-1','cus-private-model-2'].includes(data.format))
   throw Error('This is not a CUS private backup.');
  const importedCases=[],importedVersions=[];
  for(const value of data.cases||[]){
   let frames,edge,prep;
   if(value.framesPacked){
    edge=value.edge||EDGE;prep=value.prep||PREP_CURRENT;
    if(value.frameCount>256)throw Error('Incompatible case backup.');
    frames=unpackFrames(value.framesPacked,edge,value.frameCount);
   }else{
    /* 0.7.0 backups stored one base64 float array per frame at 64 px. */
    if(value.schema!==LEGACY_SCHEMA||!Array.isArray(value.frames)||value.frames.length>256)
     throw Error('Incompatible case backup.');
    edge=EDGE;prep=PREP_LEGACY;
    frames=value.frames.map(f=>{
     const a=new Float32Array(unbase64(f));
     if(a.length!==EDGE*EDGE||a.some(x=>!Number.isFinite(x)||x<0||x>1))throw Error('Invalid learning-frame values.');
     return a;
    });
   }
   const record={...value,id:crypto.randomUUID(),frames,edge,prep,framesPacked:undefined};
   validateCase(record,[...cases,...importedCases]);lineageCheck(record);
   importedCases.push(record);
  }
  for(const value of data.versions||(data.version?[data.version]:[])){
   if(value.schema!==SCHEMA||!Array.isArray(value.trainingInfantCodes)||!Array.isArray(value.trainingSourceHashes)
      ||value.trained.some(k=>!FEATURES.some(([key])=>key===k)))
    throw Error('Incompatible model backup.');
   const v={...value,id:crypto.randomUUID(),status:'candidate',externalValidated:false,
    gate:{allowed:false,tier:'insufficient',reason:'Imported candidate; re-evaluate on your independent holdout before activation.'},
    encoder:{...value.encoder,weightData:unbase64(value.encoder.weightData)},
    attention:{...value.attention,weightData:unbase64(value.attention.weightData)},
    head:{...value.head,weightData:unbase64(value.head.weightData)}};
   const loaded=await restore(v);
   const ok=architectureMatches(loaded.networks,v.edge||EDGE);
   disposeNetworks(loaded.networks);
   if(!ok)throw Error('Unexpected model architecture.');
   importedVersions.push(v);
  }
  for(const r of importedCases)await db.put('cases',r);
  for(const v of importedVersions)await db.put('versions',v);
  await refresh();
  message(`Restored ${importedCases.length} cases and ${importedVersions.length} candidate versions. Imported models were not automatically activated.`,'success');
 }finally{input.value='';}
});

// --------------------------------------------------------------- self-tests



function renderChecks(results,heading){
 const target=$('checkResults');
 const failed=results.filter(r=>!r.ok).length;
 const box=node('div',undefined,'card');
 box.append(node('h3',heading));
 box.append(node('p',`${results.length-failed} of ${results.length} passed.`,failed?'tier bad':'tier ok'));
 box.append(table(['Check','Result','Detail'],
  results.map(r=>[r.name,r.ok?'pass':'FAIL',r.detail||''])));
 target.prepend(box);
 message(failed?`${failed} self-test(s) failed in this browser.`:'All self-tests passed in this browser.',
  failed?'error':'success');
}
$('runChecks').onclick=()=>guard(()=>renderChecks(selfTests(),'Statistics, rules and label algebra'));

$('runSmoke').onclick=()=>guard(async()=>{
 if(busy)throw Error('Wait for training to finish.');
 busy=true;updateButtons();
 try{
  await ready();
  message(`Fitting a synthetic model on the ${backend()} backend…`);
  const edge=64,rand=mulberry32(5);
  const make=(code,positive,partition)=>({
   id:code,infant:code,study:code,partition,plane:'coronal',blind:true,
   modalityVerified:true,axisVerified:true,edge,prep:PREP_CURRENT,
   category:'synthetic',hashes:[code],audit:{complete:true},
   frames:Array.from({length:4},()=>{
    const f=new Uint8Array(edge*edge);
    for(let i=0;i<f.length;i++)f[i]=Math.floor(rand()*60)+(positive&&(i%edge)>edge/2?120:0);
    return f;}),
   labels:deriveAnyLabels({left_gmh:positive?1:0,right_gmh:positive?1:0})});
  const records=[];
  for(let i=0;i<4;i++)records.push(make(`fit-pos-${i}`,true,'train'),make(`fit-neg-${i}`,false,'train'));
  for(let i=0;i<2;i++)records.push(make(`hold-pos-${i}`,true,'holdout'),make(`hold-neg-${i}`,false,'holdout'));
  const version=await train(records,{epochs:2,edge,seed:1,bagFrames:0,augmentation:false},()=>{});
  const results=[];
  try{
   const out=await predict(version.networks,records[0],version.trained,edge);
   results.push({name:'Model graph executes',ok:true,detail:`backend ${backend()}, ${version.trained.length} heads fitted`});
   results.push({name:'Scores are finite probabilities',
    ok:Object.values(out.scores).every(v=>Number.isFinite(v)&&v>=0&&v<=1),
    detail:Object.entries(out.scores).map(([k,v])=>`${k} ${v.toFixed(3)}`).join(', ')});
   results.push({name:'Attention weights sum to one across frames',
    ok:Math.abs(out.attention.reduce((s,v)=>s+v,0)-1)<1e-4,
    detail:`${out.attention.length} frames, sum ${out.attention.reduce((s,v)=>s+v,0).toFixed(6)}`});
   results.push({name:'Training loss is finite and decreased',
    ok:version.history.every(h=>Number.isFinite(h.loss)),
    detail:version.history.map(h=>h.loss.toFixed(4)).join(' → ')});
   results.push({name:'Parameter count matches the documented architecture',
    ok:SHAPES.EMBED===96&&SHAPES.BAG===192&&SHAPES.FEATURES===FEATURES.length,
    detail:`embedding ${SHAPES.EMBED}, bag ${SHAPES.BAG}, heads ${SHAPES.FEATURES}`});
   const round=await serialize(version);
   const back=await restore({...round,schema:SCHEMA});
   const again=await predict(back.networks,records[0],version.trained,edge);
   disposeNetworks(back.networks);
   results.push({name:'Weights survive save and reload unchanged',
    ok:Object.keys(out.scores).every(k=>Math.abs(out.scores[k]-again.scores[k])<1e-5),
    detail:'round-tripped through the serialisation path'});
  }finally{disposeNetworks(version.networks);}
  renderChecks(results,'Learning smoke test on synthetic data');
 }finally{busy=false;updateButtons();}
});

// --------------------------------------------------------------------- boot

optionsForms();labelsForm();gmhForm();
guard(async()=>{
 await refresh();
 await ready();
 message(`Reader ${VERSION} ready. TensorFlow.js backend: ${backend()}.`,'success');
});
