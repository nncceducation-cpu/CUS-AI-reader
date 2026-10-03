/* CUS Reader media decoding and preprocessing.

   Decoding stays exhaustive: every frame of every accepted source is decoded
   in stored order, and a source over budget is rejected rather than sampled.

   Preprocessing changed in 0.8.0. Frames are letterboxed rather than stretched
   to square, stored as 8-bit at a selectable edge, optionally cropped to the
   ultrasound sector, and NIfTI intensities are windowed on robust percentiles
   instead of a divisor that collapsed low-dynamic-range volumes. */

import {EDGES, DEFAULT_EDGE, PREP_CURRENT} from './core.js';

export const MAX_FRAMES=256, MAX_BYTES=256*1024*1024;
const ANALYSIS_EDGE=256;      // working size for sector and annotation analysis
const LIT=16;                 // >6% of full scale counts as inside the sector
const SATURATED=237;          // >93% of full scale counts as burned-in bright

export async function sha256(buffer){
 return [...new Uint8Array(await crypto.subtle.digest('SHA-256',buffer))]
  .map(v=>v.toString(16).padStart(2,'0')).join('');
}

/* Aspect-preserving fit. 0.7.0 stretched every source to a square, so a
   733 x 494 coronal sweep was compressed by a factor of 1.48 in one axis and
   two scanners with different frame shapes produced differently distorted
   anatomy for the same finding. */
export function letterbox(width,height,edge){
 const scale=Math.min(edge/width,edge/height);
 const w=Math.max(1,Math.round(width*scale)),h=Math.max(1,Math.round(height*scale));
 return {w,h,x:Math.floor((edge-w)/2),y:Math.floor((edge-h)/2),scale};
}

function surface(width,height){
 const canvas=document.createElement('canvas');
 canvas.width=width;canvas.height=height;
 return {canvas,ctx:canvas.getContext('2d',{willReadFrequently:true})};
}

function greyscale(drawable,width,height,edge){
 const scale=Math.min(1,edge/Math.max(width,height));
 const aw=Math.max(1,Math.round(width*scale)),ah=Math.max(1,Math.round(height*scale));
 const {ctx}=surface(aw,ah);
 ctx.drawImage(drawable,0,0,width,height,0,0,aw,ah);
 const rgba=ctx.getImageData(0,0,aw,ah).data,grey=new Uint8Array(aw*ah);
 for(let i=0;i<grey.length;i++)grey[i]=(.299*rgba[i*4]+.587*rgba[i*4+1]+.114*rgba[i*4+2])|0;
 return {grey,aw,ah};
}

/* Bounding box of the illuminated sector, estimated from row and column
   occupancy profiles. Returns null rather than cropping when the estimate
   would discard more than three quarters of the frame, so a dark study is
   never silently reduced to a corner. */
export function sectorBox(analysis,width,height){
 const {grey,aw,ah}=analysis;
 const cols=new Int32Array(aw),rows=new Int32Array(ah);
 for(let y=0;y<ah;y++)for(let x=0;x<aw;x++)if(grey[y*aw+x]>LIT){cols[x]++;rows[y]++;}
 const span=(profile,extent)=>{
  const need=Math.max(1,Math.round(extent*.015));
  let first=-1,last=-1;
  for(let i=0;i<profile.length;i++)if(profile[i]>=need){if(first<0)first=i;last=i;}
  return [first,last];
 };
 const [x0,x1]=span(cols,ah),[y0,y1]=span(rows,aw);
 if(x0<0||y0<0)return null;
 const pad=2;
 const bx=Math.max(0,x0-pad),by=Math.max(0,y0-pad);
 const bw=Math.min(aw,x1+1+pad)-bx,bh=Math.min(ah,y1+1+pad)-by;
 if(bw<1||bh<1||bw*bh<aw*ah*.25)return null;
 const sx=width/aw,sy=height/ah;
 return {x:Math.round(bx*sx),y:Math.round(by*sy),
  w:Math.max(1,Math.round(bw*sx)),h:Math.max(1,Math.round(bh*sy)),
  fraction:(bw*bh)/(aw*ah)};
}

/* Burned-in annotation screen. This is a de-identification aid, not a
   guarantee: it measures saturated pixels in the upper band against the
   centre of the image and cannot read text or recognise a logo. */
export function annotationScan(analysis){
 const {grey,aw,ah}=analysis;
 const bandRows=Math.max(1,Math.round(ah*.1));
 let band=0;
 for(let y=0;y<bandRows;y++)for(let x=0;x<aw;x++)if(grey[y*aw+x]>SATURATED)band++;
 let centre=0,centreTotal=0;
 for(let y=Math.round(ah*.3);y<Math.round(ah*.7);y++)for(let x=0;x<aw;x++){
  centreTotal++;if(grey[y*aw+x]>SATURATED)centre++;
 }
 const bandFraction=band/(bandRows*aw),centreFraction=centreTotal?centre/centreTotal:0;
 return {bandSaturatedPixels:band,bandFraction,centreFraction,
  suspected:bandFraction>.005&&bandFraction>=centreFraction*3};
}

/* One preprocessing context per source. The sector box is estimated from the
   first frame and reused for the rest: within one acquisition the sector
   geometry is fixed, and a per-frame box would make the stored anatomy move
   between frames. */
export function createPrep(options={}){
 const edge=EDGES.includes(options.edge)?options.edge:DEFAULT_EDGE;
 const sectorCrop=options.sectorCrop!==false;
 const blankTopBand=!!options.blankTopBand;
 const {canvas,ctx}=surface(edge,edge);
 let box=null,diagnostics=null,calibrated=false;

 function calibrate(drawable,width,height){
  calibrated=true;
  const analysis=greyscale(drawable,width,height,ANALYSIS_EDGE);
  diagnostics=annotationScan(analysis);
  box=sectorCrop?sectorBox(analysis,width,height):null;
 }
 function frame(drawable,width,height){
  if(!calibrated)calibrate(drawable,width,height);
  const src=box||{x:0,y:0,w:width,h:height};
  const fit=letterbox(src.w,src.h,edge);
  ctx.fillStyle='#000';ctx.fillRect(0,0,edge,edge);
  ctx.drawImage(drawable,src.x,src.y,src.w,src.h,fit.x,fit.y,fit.w,fit.h);
  const rgba=ctx.getImageData(0,0,edge,edge).data,out=new Uint8Array(edge*edge);
  for(let i=0;i<out.length;i++)out[i]=(.299*rgba[i*4]+.587*rgba[i*4+1]+.114*rgba[i*4+2])|0;
  if(blankTopBand){
   const rows=Math.max(1,Math.round(fit.h*.1));
   for(let y=fit.y;y<Math.min(edge,fit.y+rows);y++)out.fill(0,y*edge,(y+1)*edge);
  }
  return out;
 }
 return {edge,sectorCrop,blankTopBand,calibrate,frame,
  report:()=>({edge,sectorCrop,blankTopBand,cropBox:box,annotation:diagnostics})};
}

export function preview(frame,edge){
 const side=edge||Math.round(Math.sqrt(frame.length));
 const {canvas,ctx}=surface(side,side);
 const data=ctx.createImageData(side,side),float=frame instanceof Float32Array;
 for(let i=0;i<frame.length;i++){
  const v=float?Math.round(frame[i]*255):frame[i];
  data.data[i*4]=v;data.data[i*4+1]=v;data.data[i*4+2]=v;data.data[i*4+3]=255;
 }
 ctx.putImageData(data,0,0);return canvas.toDataURL();
}

async function decompress(buffer){
 const reader=new Blob([buffer]).stream().pipeThrough(new DecompressionStream('gzip')).getReader(),chunks=[];
 let size=0;
 try{
  while(true){
   const {value,done}=await reader.read();if(done)break;
   size+=value.byteLength;
   if(size>MAX_BYTES)throw Error('Decompressed source exceeds the 256 MB browser budget. Use the desktop reader.');
   chunks.push(value);
  }
 }finally{reader.releaseLock();}
 const out=new Uint8Array(size);let offset=0;
 for(const chunk of chunks){out.set(chunk,offset);offset+=chunk.byteLength;}
 return out.buffer;
}

/* Robust display window.

   0.7.0 took one of two branches, and both lost contrast:

   - A volume already inside 0..255 was passed through UNSCALED. This is the
     branch that bit in practice: a 0 to 12 integer volume rendered into 12 of
     255 levels, and a float volume normalised to 0 to 0.3 into 0.3 of 255,
     which is essentially black.
   - Anything else was divided by Math.max(max-min,1), which loses contrast
     whenever the range is under 1: a volume spanning -0.5 to -0.25 rendered at
     25% of available scale.

   Percentile windowing covers both. */
export function windowBounds(sample,min,max){
 const sorted=Float64Array.from(sample).sort();
 const pick=q=>sorted[Math.max(0,Math.min(sorted.length-1,Math.floor(q*(sorted.length-1))))];
 let lo=pick(.01),hi=pick(.99);
 if(!(hi>lo)){lo=min;hi=max;}
 return {lo,hi,degenerate:!(hi>lo)};
}

export function niftiData(buffer){
 const v=new DataView(buffer);const little=v.getInt32(0,true)===348;
 if(!little&&v.getInt32(0,false)!==348)throw Error('Only single-file NIfTI-1 is supported in this browser.');
 if(String.fromCharCode(...new Uint8Array(buffer,344,3))!=='n+1')throw Error('NIfTI must be a single-file .nii object.');
 const dims=v.getInt16(40,little),w=v.getInt16(42,little),h=v.getInt16(44,little);
 const third=dims>=3?v.getInt16(46,little):1,fourth=dims>=4?v.getInt16(48,little):1;
 if(dims<2||dims>4||w<1||h<1||third<1||fourth<1)throw Error('Ambiguous NIfTI dimensions; resolve spatial/time axes before upload.');
 /* A 2D-plus-time cine stores its frames on the fourth axis with a singleton
    third axis. 0.7.0 rejected that shape outright. Genuinely ambiguous volumes
    with two non-singleton axes are still rejected. */
 if(dims===4&&third>1&&fourth>1)throw Error('Ambiguous NIfTI dimensions; resolve spatial/time axes before upload.');
 const n=third>1?third:fourth;
 const axis=third>1?'Stored third axis; requires confirmation':'Stored fourth (time) axis; requires confirmation';
 if(n>MAX_FRAMES)throw Error(`Source has ${n} frames; the browser budget is ${MAX_FRAMES}. Use the desktop reader; no frames were sampled.`);
 const types={2:[1,'getUint8'],4:[2,'getInt16'],8:[4,'getInt32'],16:[4,'getFloat32'],64:[8,'getFloat64'],256:[1,'getInt8'],512:[2,'getUint16'],768:[4,'getUint32']};
 const type=types[v.getInt16(70,little)];if(!type)throw Error('Unsupported NIfTI pixel datatype.');
 const offset=v.getFloat32(108,little);
 if(!Number.isInteger(offset)||offset<352||offset+w*h*n*type[0]>buffer.byteLength)throw Error('Truncated or invalid NIfTI payload.');
 const rawSlope=v.getFloat32(112,little),slope=rawSlope===0?1:rawSlope,intercept=rawSlope===0?0:v.getFloat32(116,little);
 const value=i=>v[type[1]](offset+i*type[0],little)*slope+intercept;
 const total=w*h*n,stride=Math.max(1,Math.floor(total/200000)),sample=[];
 let min=Infinity,max=-Infinity;
 for(let i=0;i<total;i++){
  const x=value(i);
  if(!Number.isFinite(x))throw Error('Non-finite NIfTI intensities.');
  if(x<min)min=x;if(x>max)max=x;
  if(i%stride===0)sample.push(x);
 }
 return {w,h,n,value,min,max,window:windowBounds(sample,min,max),axis,
  spacing:[v.getFloat32(80,little),v.getFloat32(84,little)],units:v.getUint8(123)&7,
  windowSampleStride:stride};
}

function decodeNifti(buffer,prep){
 const volume=niftiData(buffer),{w,h,n,value,min,max,window,axis}=volume;
 const {canvas,ctx}=surface(w,h);
 const rgba=ctx.createImageData(w,h),frames=[];
 const span=window.degenerate?0:window.hi-window.lo;
 for(let f=0;f<n;f++){
  for(let i=0;i<w*h;i++){
   const native=value(f*w*h+i);
   const p=span?Math.min(255,Math.max(0,(native-window.lo)/span*255)):0;
   rgba.data[i*4]=p;rgba.data[i*4+1]=p;rgba.data[i*4+2]=p;rgba.data[i*4+3]=255;
  }
  ctx.putImageData(rgba,0,0);
  frames.push(prep.frame(canvas,w,h));
 }
 return {frames,audit:{kind:'NIfTI',sourceFrames:n,decodedFrames:n,complete:true,width:w,height:h,
  axis,spacingVerified:false,intensityRange:[min,max],
  displayWindow:[window.lo,window.hi],windowSampleStride:volume.windowSampleStride},
  axisSuggested:false};
}

async function decodeImages(file,buffer,prep){
 const animated=/\.(gif|webp)$/i.test(file.name);
 if(animated){
  if(!globalThis.ImageDecoder)throw Error('This browser cannot exhaustively decode animated images. Use Chrome or Edge.');
  const decoder=new ImageDecoder({data:buffer,type:file.type||(/gif$/i.test(file.name)?'image/gif':'image/webp')});
  await decoder.tracks.ready;
  const n=decoder.tracks.selectedTrack.frameCount;
  if(!n||n>MAX_FRAMES){decoder.close();throw Error('Animated-image frame count is unknown or exceeds the browser budget.');}
  const frames=[];
  try{
   for(let i=0;i<n;i++){
    const {image}=await decoder.decode({frameIndex:i});
    frames.push(prep.frame(image,image.displayWidth,image.displayHeight));
    image.close();
   }
  }finally{decoder.close();}
  return {frames,audit:{kind:'Animated image',sourceFrames:n,decodedFrames:n,complete:true},axisSuggested:true};
 }
 const bitmap=await createImageBitmap(file);
 try{
  return {frames:[prep.frame(bitmap,bitmap.width,bitmap.height)],
   audit:{kind:'Image',sourceFrames:1,decodedFrames:1,complete:true,width:bitmap.width,height:bitmap.height},
   axisSuggested:true};
 }finally{bitmap.close();}
}

function codecDescription(file,track){
 const entry=file.getTrackById(track.id).mdia.minf.stbl.stsd.entries[0];
 const box=entry.avcC||entry.hvcC||entry.vpcC||entry.av1C;
 if(!box)return undefined;
 const stream=new globalThis.DataStream(undefined,0,globalThis.DataStream.BIG_ENDIAN);
 box.write(stream);
 return new Uint8Array(stream.buffer,8);
}

async function decodeMp4(buffer,prep){
 if(!globalThis.VideoDecoder||!globalThis.MP4Box)throw Error('MP4 decoding requires Chrome or Edge and the media library.');
 const mp4=MP4Box.createFile(),samples=[];let track,config;
 await new Promise((resolve,reject)=>{
  mp4.onError=reject;
  mp4.onReady=info=>{try{
   track=info.videoTracks[0];
   if(!track)throw Error('No video track found.');
   if(track.nb_samples>MAX_FRAMES)throw Error(`Clip exceeds the ${MAX_FRAMES}-frame browser budget. Use the desktop reader; no frames were sampled.`);
   config={codec:track.codec,codedWidth:track.video.width,codedHeight:track.video.height,description:codecDescription(mp4,track)};
   mp4.setExtractionOptions(track.id,null,{nbSamples:track.nb_samples});mp4.start();
  }catch(e){reject(e);}};
  mp4.onSamples=(_id,_user,chunk)=>{samples.push(...chunk);if(samples.length===track.nb_samples)resolve();};
  buffer.fileStart=0;
  try{mp4.appendBuffer(buffer);mp4.flush();if(track&&samples.length!==track.nb_samples)reject(Error('Incomplete sample extraction.'));}catch(e){reject(e);}
 });
 const supported=await VideoDecoder.isConfigSupported(config);
 if(!supported.supported)throw Error('Video codec is unsupported by this browser.');
 const results=[];let decodeError;
 const decoder=new VideoDecoder({
  output:frame=>{try{results.push({timestamp:frame.timestamp,pixels:prep.frame(frame,frame.displayWidth,frame.displayHeight)});}catch(e){decodeError=e;}finally{frame.close();}},
  error:e=>{decodeError=e;}});
 try{
  decoder.configure(config);
  for(const sample of samples){
   decoder.decode(new EncodedVideoChunk({type:sample.is_sync?'key':'delta',timestamp:Math.round(sample.cts*1e6/sample.timescale),duration:Math.round(sample.duration*1e6/sample.timescale),data:sample.data}));
   if(decoder.decodeQueueSize>16)await new Promise(r=>setTimeout(r,0));
  }
  await decoder.flush();
  if(decodeError)throw decodeError;
  if(results.length!==samples.length)throw Error('Incomplete clip decoding; inference is withheld.');
 }finally{decoder.close();}
 results.sort((a,b)=>a.timestamp-b.timestamp);
 return {frames:results.map(r=>r.pixels),
  audit:{kind:'MP4/MOV clip',sourceFrames:samples.length,decodedFrames:results.length,complete:true,width:track.video.width,height:track.video.height},
  axisSuggested:true};
}

export async function decodeFiles(files,onProgress=()=>{},options={}){
 const edge=EDGES.includes(options.edge)?options.edge:DEFAULT_EDGE;
 const result={frames:[],sources:[],hashes:[],edge,prep:PREP_CURRENT,
  prepOptions:{edge,sectorCrop:options.sectorCrop!==false,blankTopBand:!!options.blankTopBand},
  audit:{complete:true,sourceFrames:0,decodedFrames:0,annotationWarnings:[]},
  modalityVerified:false,axisSuggested:true};
 for(const file of files){
  if(file.size>MAX_BYTES)throw Error('A source exceeds the 256 MB browser budget. Use the desktop reader.');
  onProgress(`Decoding ${file.name}`);
  let buffer=await file.arrayBuffer();
  result.hashes.push(await sha256(buffer));
  const prep=createPrep({...options,edge});
  let decoded;
  if(/\.nii(\.gz)?$/i.test(file.name)){
   if(/\.gz$/i.test(file.name))buffer=await decompress(buffer);
   decoded=decodeNifti(buffer,prep);
  }
  else if(/\.(mp4|mov|m4v)$/i.test(file.name))decoded=await decodeMp4(buffer,prep);
  else if(/\.(png|jpe?g|bmp|gif|webp)$/i.test(file.name))decoded=await decodeImages(file,buffer,prep);
  else throw Error('Browser support: PNG, JPEG, BMP, GIF, WebP, MP4/MOV, NIfTI. Use the Python reader for DICOM, AVI, MKV and TIFF.');
  if(result.frames.length+decoded.frames.length>MAX_FRAMES)throw Error('The complete study exceeds the 256-frame browser budget. Use the desktop reader; no sampling is allowed.');
  const report=prep.report();
  result.frames.push(...decoded.frames);
  result.sources.push({name:file.name,...decoded.audit,preprocessing:report});
  result.audit.sourceFrames+=decoded.audit.sourceFrames;
  result.audit.decodedFrames+=decoded.audit.decodedFrames;
  if(report.annotation?.suspected)result.audit.annotationWarnings.push(file.name);
  result.axisSuggested&&=decoded.axisSuggested;
 }
 if(!result.frames.length)throw Error('Choose at least one supported source.');
 return result;
}
