import {EDGE} from './core.js';
const MAX_FRAMES=256,MAX_BYTES=256*1024*1024;
export async function sha256(buffer){return [...new Uint8Array(await crypto.subtle.digest('SHA-256',buffer))].map(v=>v.toString(16).padStart(2,'0')).join('');}
function surface(){const canvas=document.createElement('canvas');canvas.width=EDGE;canvas.height=EDGE;return {canvas,ctx:canvas.getContext('2d',{willReadFrequently:true})};}
function pixels(drawable,width,height){const {canvas,ctx}=surface();ctx.drawImage(drawable,0,0,width,height,0,0,EDGE,EDGE);const rgba=ctx.getImageData(0,0,EDGE,EDGE).data,out=new Float32Array(EDGE*EDGE);for(let i=0;i<out.length;i++)out[i]=(.299*rgba[i*4]+.587*rgba[i*4+1]+.114*rgba[i*4+2])/255;return out;}
export function preview(frame){const canvas=document.createElement('canvas');canvas.width=EDGE;canvas.height=EDGE;const ctx=canvas.getContext('2d'),data=ctx.createImageData(EDGE,EDGE);for(let i=0;i<frame.length;i++){const value=Math.round(frame[i]*255);data.data.set([value,value,value,255],i*4);}ctx.putImageData(data,0,0);return canvas.toDataURL();}
async function decompress(buffer){const reader=new Blob([buffer]).stream().pipeThrough(new DecompressionStream('gzip')).getReader(),chunks=[];let size=0;try{while(true){const {value,done}=await reader.read();if(done)break;size+=value.byteLength;if(size>MAX_BYTES)throw Error('Decompressed source exceeds the 256 MB browser budget. Use the desktop reader.');chunks.push(value);}}finally{reader.releaseLock();}const out=new Uint8Array(size);let offset=0;for(const chunk of chunks){out.set(chunk,offset);offset+=chunk.byteLength;}return out.buffer;}
export function niftiData(buffer){
 const v=new DataView(buffer);const little=v.getInt32(0,true)===348;
 if(!little&&v.getInt32(0,false)!==348)throw Error('Only single-file NIfTI-1 is supported in this browser.');
 if(String.fromCharCode(...new Uint8Array(buffer,344,3))!=='n+1')throw Error('NIfTI must be a single-file .nii object.');
 const dims=v.getInt16(40,little),w=v.getInt16(42,little),h=v.getInt16(44,little),n=dims>=3?v.getInt16(46,little):1;
 if(dims<2||dims>4||(dims===4&&v.getInt16(48,little)!==1)||w<1||h<1||n<1)throw Error('Ambiguous NIfTI dimensions; resolve spatial/time axes before upload.');
 if(n>MAX_FRAMES)throw Error(`Source has ${n} frames; the browser budget is ${MAX_FRAMES}. Use the desktop reader; no frames were sampled.`);
 const types={2:[1,'getUint8'],4:[2,'getInt16'],8:[4,'getInt32'],16:[4,'getFloat32'],64:[8,'getFloat64'],256:[1,'getInt8'],512:[2,'getUint16'],768:[4,'getUint32']};
 const type=types[v.getInt16(70,little)];if(!type)throw Error('Unsupported NIfTI pixel datatype.');
 const offset=v.getFloat32(108,little);if(!Number.isInteger(offset)||offset<352||offset+w*h*n*type[0]>buffer.byteLength)throw Error('Truncated or invalid NIfTI payload.');
 const rawSlope=v.getFloat32(112,little),slope=rawSlope===0?1:rawSlope,intercept=rawSlope===0?0:v.getFloat32(116,little);
 const value=i=>v[type[1]](offset+i*type[0],little)*slope+intercept;
 let min=Infinity,max=-Infinity;for(let i=0;i<w*h*n;i++){const x=value(i);if(!Number.isFinite(x))throw Error('Non-finite NIfTI intensities.');min=Math.min(min,x);max=Math.max(max,x);}
 return {w,h,n,value,min,max,spacing:[v.getFloat32(80,little),v.getFloat32(84,little)],units:v.getUint8(123)&7};
}
function decodeNifti(buffer){const volume=niftiData(buffer),frames=[],{w,h,n,value,min,max}=volume;
 const canvas=document.createElement('canvas');canvas.width=w;canvas.height=h;const ctx=canvas.getContext('2d'),rgba=ctx.createImageData(w,h);
 const rendered=min>=0&&max<=255;
 for(let f=0;f<n;f++){for(let i=0;i<w*h;i++){const native=value(f*w*h+i),p=rendered?native:(native-min)/Math.max(max-min,1)*255;rgba.data.set([p,p,p,255],i*4);}ctx.putImageData(rgba,0,0);frames.push(pixels(canvas,w,h));}
 return {frames,audit:{kind:'NIfTI',sourceFrames:n,decodedFrames:n,complete:true,width:w,height:h,
  axis:'Stored third axis; requires confirmation',spacingVerified:false,intensityRange:[min,max]},modalityVerified:false,axisVerified:false};
}
async function decodeImages(file,buffer){
 const animated=/\.(gif|webp)$/i.test(file.name);
 if(animated){
  if(!globalThis.ImageDecoder)throw Error('This browser cannot exhaustively decode animated images. Use Chrome or Edge.');
  const decoder=new ImageDecoder({data:buffer,type:file.type||(/gif$/i.test(file.name)?'image/gif':'image/webp')});await decoder.tracks.ready;
  const n=decoder.tracks.selectedTrack.frameCount;if(!n||n>MAX_FRAMES){decoder.close();throw Error('Animated-image frame count is unknown or exceeds the browser budget.');}
  const frames=[];try{for(let i=0;i<n;i++){const {image}=await decoder.decode({frameIndex:i});frames.push(pixels(image,image.displayWidth,image.displayHeight));image.close();}}finally{decoder.close();}
  return {frames,audit:{kind:'Animated image',sourceFrames:n,decodedFrames:n,complete:true},modalityVerified:false,axisVerified:true};
 }
 const bitmap=await createImageBitmap(file);try{return {frames:[pixels(bitmap,bitmap.width,bitmap.height)],audit:{kind:'Image',sourceFrames:1,decodedFrames:1,complete:true,width:bitmap.width,height:bitmap.height},modalityVerified:false,axisVerified:true};}finally{bitmap.close();}
}
function codecDescription(file,track){const entry=file.getTrackById(track.id).mdia.minf.stbl.stsd.entries[0];const box=entry.avcC||entry.hvcC||entry.vpcC||entry.av1C;if(!box)return undefined;const stream=new globalThis.DataStream(undefined,0,globalThis.DataStream.BIG_ENDIAN);box.write(stream);return new Uint8Array(stream.buffer,8);}
async function decodeMp4(buffer){
 if(!globalThis.VideoDecoder||!globalThis.MP4Box)throw Error('MP4 decoding requires Chrome or Edge and the media library.');
 const mp4=MP4Box.createFile(),samples=[];let track,config;
 await new Promise((resolve,reject)=>{
  mp4.onError=reject;
  mp4.onReady=info=>{try{track=info.videoTracks[0];if(!track)throw Error('No video track found.');if(track.nb_samples>MAX_FRAMES)throw Error(`Clip exceeds the ${MAX_FRAMES}-frame browser budget. Use the desktop reader; no frames were sampled.`);
   config={codec:track.codec,codedWidth:track.video.width,codedHeight:track.video.height,description:codecDescription(mp4,track)};
   mp4.setExtractionOptions(track.id,null,{nbSamples:track.nb_samples});mp4.start();}catch(e){reject(e);}};
  mp4.onSamples=(_id,_user,chunk)=>{samples.push(...chunk);if(samples.length===track.nb_samples)resolve();};
  buffer.fileStart=0;try{mp4.appendBuffer(buffer);mp4.flush();if(track&&samples.length!==track.nb_samples)reject(Error('Incomplete sample extraction.'));}catch(e){reject(e);}
 });
 const supported=await VideoDecoder.isConfigSupported(config);if(!supported.supported)throw Error('Video codec is unsupported by this browser.');
 const results=[];let decodeError;
 const decoder=new VideoDecoder({output:frame=>{try{results.push({timestamp:frame.timestamp,pixels:pixels(frame,frame.displayWidth,frame.displayHeight)});}catch(e){decodeError=e;}finally{frame.close();}},error:e=>{decodeError=e;}});
 try{decoder.configure(config);for(const sample of samples){decoder.decode(new EncodedVideoChunk({type:sample.is_sync?'key':'delta',timestamp:Math.round(sample.cts*1e6/sample.timescale),duration:Math.round(sample.duration*1e6/sample.timescale),data:sample.data}));if(decoder.decodeQueueSize>16)await new Promise(r=>setTimeout(r,0));}await decoder.flush();if(decodeError)throw decodeError;if(results.length!==samples.length)throw Error('Incomplete clip decoding; inference is withheld.');}finally{decoder.close();}
 results.sort((a,b)=>a.timestamp-b.timestamp);
 return {frames:results.map(r=>r.pixels),audit:{kind:'MP4/MOV clip',sourceFrames:samples.length,decodedFrames:results.length,complete:true,width:track.video.width,height:track.video.height},modalityVerified:false,axisVerified:true};
}
export async function decodeFiles(files,onProgress=()=>{}){
 const result={frames:[],sources:[],hashes:[],audit:{complete:true,sourceFrames:0,decodedFrames:0},modalityVerified:false,axisVerified:true};
 for(const file of files){if(file.size>MAX_BYTES)throw Error('A source exceeds the 256 MB browser budget. Use the desktop reader.');onProgress(`Decoding ${file.name}`);let buffer=await file.arrayBuffer();result.hashes.push(await sha256(buffer));let decoded;
  if(/\.nii(\.gz)?$/i.test(file.name)){if(/\.gz$/i.test(file.name))buffer=await decompress(buffer);decoded=decodeNifti(buffer);}
  else if(/\.(mp4|mov|m4v)$/i.test(file.name))decoded=await decodeMp4(buffer);
  else if(/\.(png|jpe?g|bmp|gif|webp)$/i.test(file.name))decoded=await decodeImages(file,buffer);
  else throw Error('Browser support: PNG, JPEG, BMP, GIF, WebP, MP4/MOV, NIfTI. Use the Python reader for DICOM, AVI, MKV and TIFF.');
  if(result.frames.length+decoded.frames.length>MAX_FRAMES)throw Error('The complete study exceeds the 256-frame browser budget. Use the desktop reader; no sampling is allowed.');
  result.frames.push(...decoded.frames);result.sources.push({name:file.name,...decoded.audit});result.audit.sourceFrames+=decoded.audit.sourceFrames;result.audit.decodedFrames+=decoded.audit.decodedFrames;result.axisVerified&&=decoded.axisVerified;
 }
 if(!result.frames.length)throw Error('Choose at least one supported source.');return result;
}
