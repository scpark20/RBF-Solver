'use strict';
(() => {
const $=id=>document.getElementById(id);
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=(v,d=0)=>typeof v==='number'&&Number.isFinite(v)?v.toLocaleString('ko-KR',{minimumFractionDigits:d,maximumFractionDigits:d}):'—';
const finite=v=>typeof v==='number'&&Number.isFinite(v);
const cfgLabel=c=>c===0?'0.0 · Unconditional':String(c);
const method=m=>({dpmpp:'DPM-Solver++',unipc:'UniPC',rbf:'RBF'}[m]||m);
const stateName=s=>({complete:'완료',pending:'대기',not_started:'실행 전',running:'실행 중',tuning:'배치 탐색',evaluating:'FID 평가 중',starting:'시작 중',loading:'모델 로딩',teacher:'Target 생성 중',training:'학습 중',averaging:'평균 계산 중',merging:'Target 병합 중',paused:'중단됨',error:'오류',awaiting_configuration:'설정 대기',not_implemented:'실행 경로 미구현'}[s]||s||'기록 없음');
const badge=s=>`<span class="badge ${esc(s)}">${esc(stateName(s))}</span>`;
const duration=s=>!finite(s)?'—':s<60?`${num(s,1)}초`:`${num(Math.floor(s/60))}분 ${num(s%60,0)}초`;
const yes=v=>v===true?'사용':v===false?'사용 안 함':'기록 없음';
const html=(id,v)=>{if($(id).innerHTML!==v)$(id).innerHTML=v};
const text=(id,v)=>{if($(id).textContent!==String(v))$(id).textContent=v};
const kv=rows=>`<dl class="kv">${rows.map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(v??'기록 없음')}</dd>`).join('')}</dl>`;
const progress=(done,total)=>`<div class="progress" role="progressbar" aria-label="진행률" aria-valuenow="${finite(done)?done:0}" aria-valuemin="0" aria-valuemax="${total||1}"><span style="width:${total?Math.min(100,Math.max(0,done/total*100)):0}%"></span></div>`;
const table=(headers,rows)=>`<table><thead><tr>${headers.map(h=>`<th scope="col">${h}</th>`).join('')}</tr></thead><tbody>${rows.join('')}</tbody></table>`;
let coeffs=null,coefficientNFE='all',rbfCFG=1.5;
let compare=null,rbf=null,cfg=null,nfe=null,busy=false,settingsKey='',selectorKey='';
let errors={compare:null,rbf:null,coefficients:null},view='comparison';
const scrollPositions={};
const timeText=v=>v?new Date(v*1000).toLocaleTimeString('ko-KR',{hour12:false}):'기록 없음';
function setView(v){
 scrollPositions[view]=window.scrollY;
 view=v;
 document.querySelectorAll('[data-view]').forEach(b=>{if(b.dataset.view===v)b.setAttribute('aria-current','page');else b.removeAttribute('aria-current')});
 ['comparison','rbf','gpu','settings'].forEach(k=>$(`view-${k}`).hidden=k!==v);
 window.scrollTo({top:scrollPositions[v]||0,behavior:'instant'});
}
document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>setView(b.dataset.view)));
function controls(){
 if(!compare.protocol.cfgs.includes(cfg))cfg=compare.protocol.cfgs[0];
 const p={...compare.protocol,nfes:[...new Set(compare.items.filter(i=>i.cfg===cfg).map(i=>i.nfe))].sort((a,b)=>a-b)},key=JSON.stringify([cfg,p.cfgs,p.nfes]);
 if(!p.cfgs.includes(cfg))cfg=p.cfgs[0]; if(!p.nfes.includes(nfe))nfe=p.nfes[0];
 if(key!==selectorKey){
  html('cfg-controls',p.cfgs.map(c=>`<button data-cfg="${c}" aria-pressed="${c===cfg}">CFG ${cfgLabel(c)}</button>`).join(''));
  html('nfe-controls',p.nfes.map(n=>`<button data-nfe="${n}" aria-pressed="${n===nfe}">NFE ${n}</button>`).join(''));
  $('cfg-controls').querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{cfg=Number(b.dataset.cfg);controls();renderSelected()}));
  $('nfe-controls').querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{nfe=Number(b.dataset.nfe);renderSelected()}));
  selectorKey=key;
 }
}
function find(c,m,n){return compare.items.find(i=>i.cfg===c&&i.method===m&&i.nfe===n)}
function fidCell(i,best){
 const value=i&&finite(i.fid)?num(i.fid,3):badge(i?.state||'pending');
 const prior=i?.retrained?'<div class="subtle">새 target 재학습 · 이전 '+num(i.previous?.fid,3)+'</div>':'';
 return '<td class="'+(best?'best':'')+'">'+value+prior+'</td>';
}
function chart(items,nfes){
 const values=items.filter(i=>finite(i.fid));
 if(!values.length)return '<div class="empty">FID 결과가 아직 없습니다.<br>각 조건의 평가가 완료되면 표시됩니다.</div>';
 const W=540,H=252,L=46,R=24,T=28,B=36;
 const lo=Math.min(...values.map(i=>i.fid)),hi=Math.max(...values.map(i=>i.fid));
 const pad=Math.max((hi-lo)*.22,.5),min=Math.max(0,Math.floor(lo-pad)),max=Math.ceil(hi+pad),span=max-min||1;
 const x=n=>L+(n-nfes[0])/(nfes.at(-1)-nfes[0]||1)*(W-L-R), y=f=>T+(max-f)/span*(H-T-B);
 let out=`<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="CFG ${cfg}, FID와 NFE 비교 그래프. 정확한 수치는 옆 비교 표에 표시됩니다.">`;
 for(let j=0;j<=4;j++){const f=min+span*j/4,yy=y(f);out+=`<line x1="${L}" y1="${yy}" x2="${W-R}" y2="${yy}" stroke="#e9edf3"/><text x="${L-11}" y="${yy+4}" text-anchor="end">${num(f,1)}</text>`}
 for(const n of nfes)out+=`<text x="${x(n)}" y="${H-16}" text-anchor="middle">${n}</text>`;
 out+=`<text x="${L}" y="12">FID</text><text x="${W-R}" y="${H-1}" text-anchor="end">NFE</text>`;
 for(const m of compare.protocol.methods){
  const color={dpmpp:'#315fd5',unipc:'#087f80',rbf:'#9850bd'}[m]||'#526579';
  const selected=nfes.map(n=>items.find(i=>i.method===m&&i.nfe===n));
  let segment=[]; const segments=[];
  for(const i of selected){if(i&&finite(i.fid))segment.push(i);else {if(segment.length)segments.push(segment);segment=[]}}
  if(segment.length)segments.push(segment);
  for(const seg of segments)out+=`<polyline points="${seg.map(i=>`${x(i.nfe)},${y(i.fid)}`).join(' ')}" fill="none" stroke="${color}" stroke-width="2.5" stroke-linejoin="round"/>`;
  for(const i of selected.filter(i=>i&&finite(i.fid)))out+=`<circle cx="${x(i.nfe)}" cy="${y(i.fid)}" r="4" fill="white" stroke="${color}" stroke-width="2"><title>${method(m)} · NFE ${i.nfe} · FID ${num(i.fid,3)}</title></circle>`;
 }
 return out+'</svg>';
}
function renderSelected(){
 if(!compare)return; const items=compare.items.filter(i=>i.cfg===cfg),p={...compare.protocol,nfes:[...new Set(items.map(i=>i.nfe))].sort((a,b)=>a-b)};
 $('cfg-controls').querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',Number(b.dataset.cfg)===cfg));
 $('nfe-controls').querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',Number(b.dataset.nfe)===nfe));
 html('fid-chart',chart(items,p.nfes));
 text('chart-note',`CFG ${cfgLabel(cfg)} · 조건당 ${num(p.num_samples)}장 · Y축은 선택 결과에 맞춰 조정됩니다.${items.some(i=>i.retrained)?' · NFE 6 RBF: 새 target 재학습 결과':''}`);
 html('fid-table',table(['NFE',...p.methods.map(method),'Δ RBF'],p.nfes.map(n=>{const rows=p.methods.map(m=>find(cfg,m,n)),values=rows.filter(i=>finite(i?.fid)),best=values.length?Math.min(...values.map(i=>i.fid)):null,a=find(cfg,'dpmpp',n),b=find(cfg,'unipc',n),r=find(cfg,'rbf',n),ready=[a,b,r].every(i=>finite(i?.fid)),delta=ready?r.fid-Math.min(a.fid,b.fid):null;return `<tr><th scope="row">${n}</th>${rows.map(i=>fidCell(i,finite(i?.fid)&&i.fid===best)).join('')}<td class="${ready&&delta<0?'delta':''}">${ready?`${delta>0?'+':''}${num(delta,3)}`:'—'}</td></tr>`})));
 text('cfg-summary',`선택 CFG: ${items.filter(i=>i.state==='complete').length}/${items.length}개 조건 완료 · seed ${p.seed} · ${p.order}차`);
 text('preview-description',`CFG ${cfgLabel(cfg)} · NFE ${nfe} · 저장된 첫 샘플 모음 · 선택하면 확대`);
 html('previews',p.methods.map(m=>{const i=find(cfg,m,nfe);return `<div class="preview-card"><div class="preview-head"><strong>${esc(method(m))}</strong><span class="subtle">FID ${num(i?.fid,3)}${i?.retrained?' · 재학습':''}</span></div>${i?.preview?`<button class="preview-button" data-preview="${esc(i.preview)}" data-label="${esc(method(m))} · CFG ${cfg} · NFE ${nfe}" aria-label="${esc(method(m))} CFG ${cfg} NFE ${nfe} 이미지 확대"><img src="${esc(i.preview)}" alt="${esc(method(m))}의 실제 생성 샘플" loading="lazy"></button>`:`<div class="preview-placeholder">${i?.state==='error'?'생성 오류 · 상세 기록 확인':'저장된 미리보기 없음'}</div>`}</div>`}).join(''));
}
function renderCompare(){
 const p=compare.protocol;controls();
 const finished=compare.items.filter(i=>i.state==='complete'&&finite(i.fid)),best=finished.length?finished.reduce((a,b)=>a.fid<b.fid?a:b):null;
 const pipeline=rbf?.selection?.state==='running'?rbf.selection:null;
 const stageLabel=pipeline?`RBF ${({target:'Target 생성',fit:'계수학습',sample:'샘플링'})[pipeline.stage]||pipeline.stage} · CFG ${pipeline.cfgs.join(' · ')}`:'';
 const state=compare.completed===compare.conditions?'complete':compare.items.some(i=>i.state==='error')?'error':compare.running||pipeline?'running':compare.run_state;
 html('comparison-state',badge(state));
 html('comparison-summary',`<div class="metric"><div class="metric-label">완료 조건</div><div class="metric-value">${num(compare.completed)} <small>/ ${num(compare.conditions)}</small></div><div class="metric-detail">${p.cfgs.length} CFG · ${p.methods.length} 샘플러 · CFG별 NFE 범위 적용</div></div><div class="metric"><div class="metric-label">생성 이미지</div><div class="metric-value">${num(compare.done)} <small>장</small></div><div class="metric-detail">전체 ${num(compare.total)}장 · ${finite(compare.eta_seconds)&&compare.eta_seconds>0?`예상 잔여 ${duration(compare.eta_seconds)}`:state==='complete'?'모든 조건 평가 완료':compare.running?'샘플링 실행 중':pipeline?stageLabel:'현재 실행 없음'}</div></div><div class="metric"><div class="metric-label">완료 결과 중 최저 FID</div><div class="metric-value accent">${num(best?.fid,3)}</div><div class="metric-detail">${best?`${esc(method(best.method))} · CFG ${best.cfg} · NFE ${best.nfe}`:'평가 완료 후 표시'}</div></div>`);
 const active=compare.items.filter(i=>['running','tuning','evaluating','starting','loading'].includes(i.state));
 $('active-jobs').hidden=!active.length&&!pipeline;
 if(!active.length&&pipeline)html('active-jobs',`<div class="preview-head"><strong>${esc(stageLabel)}</strong>${badge('running')}</div><p class="footnote">Target 생성 → 계수학습 → RBF 샘플링 순서로 실행 중입니다. 단계별 실제 진행은 RBF 파이프라인에서 확인할 수 있습니다.</p>`);
 if(active.length)html('active-jobs',`<div class="panel-head"><h2>현재 실행 중</h2><span class="subtle">${active.length}개 조건</span></div><div class="columns">${active.map(i=>`<div><div class="preview-head"><strong>${esc(method(i.method))} · CFG ${i.cfg} · NFE ${i.nfe}</strong>${badge(i.state)}</div>${progress(i.done,i.total)}<p class="footnote">${num(i.done)} / ${num(i.total)}장 · GPU ${num(i.gpu)} · 배치 ${num(i.batch_size)} · ${num(i.images_per_second,2)}장/초<br>처리 시간 ${duration(i.seconds)} · 마지막 기록 ${timeText(i.updated_at)}</p></div>`).join('')}</div>`);
 text('result-records',JSON.stringify(compare.items.map(({result,...i})=>({...i,result})),null,2));
 const problems=compare.items.filter(i=>['error','paused'].includes(i.state));
 $('job-alert').hidden=!problems.length;
 if(problems.length)html('job-alert',problems.map(i=>`${esc(method(i.method))} · CFG ${i.cfg} · NFE ${i.nfe}: ${esc(stateName(i.state))}${i.error?' — '+esc(i.error):''}`).join('<br>'));
 text('all-results-count',`${compare.conditions}개 조건 · FID / 진행 / 시간 / 배치 / GPU`);
 html('results-table',table(['CFG','샘플러','NFE','상태','생성 수','FID','처리 시간','장/초','배치','GPU'],compare.items.map(i=>`<tr><th scope="row">${i.cfg}</th><td>${esc(method(i.method))}${i.retrained?' · 새 target 재학습':''}</td><td>${i.nfe}</td><td>${badge(i.state)}${i.error?`<div class="error-text">${esc(i.error)}</div>`:''}</td><td>${num(i.done)} / ${num(i.total)}</td><td>${num(i.fid,3)}</td><td>${i.seconds?duration(i.seconds):'—'}</td><td>${num(i.images_per_second,2)}</td><td>${num(i.batch_size)}</td><td>${num(i.gpu)}</td></tr>`)));
 renderSelected();renderGPU();renderSettings();
}
function aggregate(rows){if(rows.length&&rows.every(i=>i.state==='complete'))return 'complete';if(rows.some(i=>i.state==='error'))return 'error';if(rows.some(i=>['teacher','training','averaging','merging','tuning','running','loading'].includes(i.state)))return rows.find(i=>['teacher','training','averaging','merging','tuning','running','loading'].includes(i.state)).state;if(rows.some(i=>i.state==='paused'))return 'paused';return 'not_started'}
function renderRBF(){
 const scope=[rbfCFG];
 html('rbf-cfg-controls',rbf.protocol.cfgs.map(c=>`<button data-rbf-cfg="${c}" aria-pressed="${c===rbfCFG}">CFG ${cfgLabel(c)}</button>`).join(''));
 $('rbf-cfg-controls').querySelectorAll('button').forEach(b=>b.onclick=()=>{rbfCFG=Number(b.dataset.rbfCfg);renderRBF();renderCoefficients();refresh()});
 const selection=rbf.selection;
 text('rbf-execution',selection?`실행 범위 CFG ${selection.cfgs.join(' · ')} · ${({target:'Target 생성',fit:'계수학습',sample:'RBF 샘플링'})[selection.stage]||selection.stage} · ${stateName(selection.state)}`:'실행 기록 없음');
 const p={...rbf.protocol,cfgs:scope,nfes:[...new Set(rbf.items.filter(i=>scope.includes(i.cfg)).map(i=>i.nfe))].sort((a,b)=>a-b)},targets=rbf.targets.filter(i=>scope.includes(i.cfg)),fits=rbf.items.filter(i=>scope.includes(i.cfg));
 const sampling=rbf.sampling,rows=(sampling?.items||[]).filter(i=>scope.includes(i.cfg)),targetDone=targets.reduce((s,i)=>s+(i.done||0),0),targetTotal=p.target_pairs*scope.length,fitDone=fits.reduce((s,i)=>s+(i.done||0),0),fitTotal=p.repetitions*fits.length;
 const targetState=aggregate(targets),fitState=aggregate(fits),sampleState=rows.length?aggregate(rows):'not_started',targetCompleted=targets.filter(i=>i.state==='complete').length,fitCompleted=fits.filter(i=>i.state==='complete').length;
 html('rbf-state',`<span class="subtle">선택 CFG ${rbfCFG}</span> ${badge(sampleState==='complete'?'complete':aggregate([...targets,...fits,...rows]))}`);
 const issues=[...targets,...fits,...rows].filter(i=>i.state==='error'||i.state==='paused');
 $('rbf-alert').hidden=!issues.length&&!rbf.selection?.error;
 if(!$('rbf-alert').hidden)html('rbf-alert',[rbf.selection?.error?esc(rbf.selection.error):'',...issues.map(i=>`CFG ${i.cfg}${i.nfe?' · NFE '+i.nfe:''}: ${esc(stateName(i.state))}${i.error?' — '+esc(i.error):''}`)].filter(Boolean).join('<br>'));
 html('rbf-stages',`<article class="panel stage-card"><div class="stage-top"><span class="stage-num">STAGE 01</span>${badge(targetState)}</div><h2>Target sampling</h2><p>${esc(method(p.teacher.method))} teacher · NFE ${p.teacher.nfe}</p><div class="stage-value">${num(targetDone)} <small>/ ${num(targetTotal)}쌍</small></div>${progress(targetDone,targetTotal)}<p>CFG ${scope.join(' · ')} · 두 GPU에 샘플 ID 분할</p><div class="stage-footer"><b>산출물</b> 노이즈·클래스·teacher target<br><b>완료</b> ${targetCompleted}/${scope.length} CFG · 병합·검증 포함</div></article><article class="panel stage-card"><div class="stage-top"><span class="stage-num">STAGE 02</span>${badge(fitState)}</div><h2>계수학습</h2><p>NFE ${p.nfes.join(' · ')} · 논문 배치 ${p.optimization_batch}쌍</p><div class="stage-value">${num(fitDone)} <small>/ ${num(fitTotal)}회</small></div>${progress(fitDone,fitTotal)}<p>${fitCompleted}/${fits.length}개 조건 완료 · 조건당 ${p.repetitions}회</p><div class="stage-footer"><b>선행 입력</b> ${targetCompleted===targets.length?'Target 생성 완료':'Target 쌍 필요'}<br><b>산출물</b> 반복별 log γ와 평균 파라미터</div></article><article class="panel stage-card"><div class="stage-top"><span class="stage-num">STAGE 03</span>${badge(sampleState)}</div><h2>RBF sampling</h2><p>학습한 파라미터 · GPU FID 평가</p><div class="stage-value">${num(rows.reduce((s,i)=>s+(i.done||0),0))} <small>/ ${num(rows.reduce((s,i)=>s+(i.total||0),0))}장</small></div>${progress(rows.reduce((s,i)=>s+(i.done||0),0),rows.reduce((s,i)=>s+(i.total||0),0)||1)}<p>${rows.filter(i=>i.state==='complete').length}/${rows.length}개 조건 완료 · 조건당 ${num(sampling?.protocol.num_samples)}장</p><div class="stage-footer"><b>선행 입력</b> 검증된 학습 파라미터<br><b>비교 입력</b> DPM++·UniPC와 동일 노이즈·클래스</div></article>`);
 text('target-description',`CFG별 ${p.target_pairs}쌍 · ${method(p.teacher.method)}-${p.teacher.nfe} teacher`);
 html('target-table',table(['CFG','생성 쌍','상태','GPU'],targets.map(i=>`<tr><th scope="row">${i.cfg}</th><td>${num(i.done)} / ${num(i.total)}</td><td>${badge(i.state)}</td><td>${i.shards?i.shards.map(r=>num(r.gpu)).join(' · '):i.gpus?.join(' · ')||num(i.gpu)}</td></tr>`)));
 text('fit-description',`각 셀은 완료 반복 / ${p.repetitions}회 · 원본 최적화 함수 재사용`);
 html('fit-table',table(['CFG',...p.nfes.map(n=>'NFE '+n)],scope.map(c=>`<tr><th scope="row">${c}</th>${p.nfes.map(n=>{const i=fits.find(i=>i.cfg===c&&i.nfe===n);return `<td>${num(i?.done)} / ${num(i?.total)}<br>${badge(i?.state||'pending')}</td>`}).join('')}</tr>`)));
 const previews=rows.filter(i=>i.preview);
 html('rbf-sampling-info',`<div class="panel-head"><div><h2>RBF 샘플링 결과</h2><p>CFG ${scope.join(' · ')} · 같은 NFE의 기존 결과와 비교 · FID는 낮을수록 좋음</p></div>${badge(sampleState)}</div><div class="table-wrap">${table(['CFG','NFE','상태','생성 수','RBF FID','DPM++ FID','UniPC FID','처리 시간','배치','GPU'],rows.map(i=>`<tr><th scope="row">${i.cfg}</th><td>${i.nfe}</td><td>${badge(i.state)}</td><td>${num(i.done)} / ${num(i.total)}</td><td>${num(i.fid,3)}</td><td>${num(compare?.items.find(r=>r.cfg===i.cfg&&r.nfe===i.nfe&&r.method==='dpmpp')?.fid,3)}</td><td>${num(compare?.items.find(r=>r.cfg===i.cfg&&r.nfe===i.nfe&&r.method==='unipc')?.fid,3)}</td><td>${i.seconds?duration(i.seconds):'—'}</td><td>${num(i.batch_size)}</td><td>${num(i.gpu)}</td></tr>`))}</div><p class="footnote">Target → 계수학습 → 샘플링 순서로 실행합니다. 각 단계의 독립 실행은 선행 결과를 검증하며, 없는 결과를 자동 생성하지 않습니다. 시간은 기존 비교와 같은 샘플링·VAE·GPU feature 통계 범위입니다.</p>${previews.length?`<div class="preview-grid" style="margin-top:20px">${previews.map(i=>`<div><div class="preview-head"><strong>RBF · CFG ${i.cfg} · NFE ${i.nfe}</strong><span class="subtle">FID ${num(i.fid,3)}</span></div><button class="preview-button" data-preview="${esc(i.preview)}" data-label="RBF · CFG ${i.cfg} · NFE ${i.nfe}" aria-label="RBF NFE ${i.nfe} 이미지 확대"><img loading="lazy" src="${esc(i.preview)}" alt="RBF NFE ${i.nfe} 생성 이미지"></button></div>`).join('')}</div>`:''}`);
 text('rbf-records',JSON.stringify({selection:rbf.selection,targets,conditions:fits,workers:rbf.gpu_workers,sampling},null,2));
 if(compare)renderCompare();renderCoefficients();renderGPU();renderSettings();
}
function workerText(w,kind){if(!w)return '기록 없음';let label=stateName(w.state);if(w.state==='complete')label='배정 작업 완료';if(w.job)label+=' · '+w.job;if(finite(w.repeat))label+=` · 반복 ${w.repeat}/${w.repetitions??'—'}`;if(finite(w.step))label+=` · 단계 ${w.step}/${w.nfe??'—'}`;return label}
function renderGPU(){
 if(!compare)return;
 html('gpu-cards',compare.gpus.length?compare.gpus.map(g=>{const c=compare.gpu_workers.find(w=>w.gpu===g.index),r=(rbf?.selection?.stage==='sample'?rbf?.sampling?.gpu_workers:rbf?.gpu_workers)?.find(w=>w.gpu===g.index);return `<article class="panel"><div class="panel-head"><div><h2 class="gpu-title">GPU ${g.index}</h2><p>${esc(g.name)}</p></div><span class="tag">${g.util===0?'현재 유휴':'사용 중'}</span></div><div class="metric-label">GPU 사용률</div><div class="gpu-big">${num(g.util)}<small> %</small></div>${progress(g.util,100)}<div class="gpu-stats"><div><span>메모리 / 전체</span><strong>${num(g.used/1024,2)} / ${num(g.total/1024,2)} GiB</strong></div><div><span>온도</span><strong>${num(g.temperature)} °C</strong></div><div><span>전력</span><strong>${num(g.power,1)} W</strong></div></div><div class="worker-line"><span class="muted">DPM++ · UniPC</span><span>${esc(workerText(c))}</span></div><div class="worker-line"><span class="muted">RBF 파이프라인</span><span>${rbf?esc(workerText(r)):'RBF 상태 불러오는 중'}</span></div>${r?.error?`<p class="error-text">${esc(r.error)}</p>`:''}</article>`}).join(''):'<div class="panel empty">GPU 조회 결과를 가져오지 못했습니다. 사용량을 0으로 추정하지 않습니다.</div>');
 const p=compare.protocol,batches=compare.items.filter(i=>finite(i.batch_size));
 html('gpu-policy',kv([['GPU 사용',p.gpus.map(g=>'GPU '+g).join(' · ')],['샘플링 배치',p.cfgs.includes(0)?'CFG 0.0: 기존 성공 배치 재사용 · 실제 OOM 시 조정':p.batch_policy==='measured_max_per_gpu'?'기존 조건의 최대 배치 실측 기록':p.batch_policy],['현재 측정 기록',batches.length?`${batches.length}개 조건 · ${num(Math.min(...batches.map(i=>i.batch_size)))}–${num(Math.max(...batches.map(i=>i.batch_size)))}장`:'측정 기록 없음'],['비교 작업 배정',p.gpu_scheduling==='shared_work_queue'?'공용 작업 대기열':p.gpu_scheduling],['RBF 작업 배정',rbf?.protocol.gpu_policy==='shared_cfg_queue'?'CFG 공용 작업 대기열':rbf?.protocol.gpu_policy],['입력 일치','샘플 ID별 동일 초기 노이즈·클래스']])+`<p class="footnote">조건별 실제 배치는 비교 결과의 전체 조건 상세에서 확인할 수 있습니다. GPU 사용률은 조회 시점의 값이며, 처리량은 해당 조건의 기록된 처리 시간 기준입니다.</p>`);
}
const sourceNames={cfgs:'CFG',methods_nfes:'샘플러와 NFE',seed:'시드',batch_size:'배치 정책',precision:'정밀도',order_num_samples:'차수와 평가 규모',time_interval:'시간 구간',reference:'ImageNet 256 참조',sampler:'샘플러 구현',rbf:'초기 RBF 범위 기록',gpu_usage:'GPU 사용',input_pairing:'입력 일치',adapter:'SiT 변환',target_pairs:'Target 수',optimization:'최적화 조건',aggregation:'반복 평균',teacher_order:'Teacher 차수',cfgs_nfes_history:'CFG·NFE·차수',adams:'Adams 분기',target_path:'Target 경로'};
function sourceList(s){return Object.entries(s||{}).map(([k,v])=>`<div class="source-item"><strong>${esc(sourceNames[k]||k)}</strong>${esc(v)}</div>`).join('')}
function renderSettings(){
 if(!compare)return; const p=compare.protocol,r=rbf?.protocol,f=compare.fid_parity,key=JSON.stringify([p,r,f]);if(key===settingsKey)return;
 const open=Array.from($('settings-content').querySelectorAll('details[open]')).map(d=>d.id);
 let out=`<div class="columns"><div class="panel"><div class="panel-head"><h2>샘플러 비교</h2></div>${kv([['모델','SiT-S/2 · ImageNet 256'],['CFG',p.cfgs.map(cfgLabel).join(' / ')],['CFG 0.0','null label만 사용 · 모든 잠재 채널 무조건부'],['샘플러',p.methods.map(method).join(' · ')],['CFG별 NFE',p.cfgs.map(c=>'CFG '+c+': '+[...new Set(compare.items.filter(i=>i.cfg===c).map(i=>i.nfe))].sort((a,b)=>a-b).join(' · ')).join(' / ')],['조건당 이미지',num(p.num_samples)+'장'],['차수',p.order+'차'],['시드',p.seed],['정밀도',p.precision],['시간 격자',p.skip_type==='time_uniform'?'균일 시간 간격':p.skip_type],['Solver 시간',`${p.t_start} → ${p.t_end}`],['SiT 물리 시간',`${1-p.t_start} → ${1-p.t_end}`],['마지막 저차 단계',yes(p.lower_order_final)],['추가 denoise',yes(p.denoise_to_zero)]])}</div><div class="panel"><div class="panel-head"><h2>ImageNet FID 평가</h2>${f?badge(f.passed?'complete':'error'):''}</div>${kv([['참조 데이터',p.reference],['Feature 추출','GPU · Inception FP32'],['통계 누적','FP64'],['검증 범위','ADM CPU feature와 GPU feature 비교'],['호환성 검사',f?(f.passed?'통과':'실패'):'기록 없음'],['검증 이미지',f?num(f.n)+'장':null],['최대 절대 오차',f?.max_abs?.toExponential(3)],['RMSE',f?.rmse?.toExponential(3)],['허용 atol / rtol',f?`${f.atol} / ${f.rtol}`:null]])}<p class="footnote">호환성 검사는 feature 구현의 일치 확인입니다. 적은 검증 이미지 수가 FID 평가 샘플 수를 뜻하지 않습니다.</p></div></div>`;
 if(r)out+=`<div class="columns"><div class="panel"><div class="panel-head"><h2>Target sampling 조건</h2></div>${kv([['CFG',r.cfgs.join(' · ')],['CFG당 target',r.target_pairs+'쌍'],['Teacher',`${method(r.teacher.method)} ${r.teacher.variant.toUpperCase()} · NFE ${r.teacher.nfe}`],['Teacher 차수',r.teacher.order+'차'],['Solver 시간',`${r.teacher.t_start} → ${r.teacher.t_end}`],['SiT 물리 시간',`${r.flow_time_start} → ${r.flow_time_end}`],['마지막 저차 단계',yes(r.teacher.lower_order_final)],['추가 denoise',yes(r.teacher.denoise_to_zero)],['시드 / 정밀도',`${r.seed} / ${r.precision}`]])}</div><div class="panel"><div class="panel-head"><h2>계수학습 조건</h2></div>${kv([['NFE',r.nfes.join(' · ')],['차수',r.history_size],['최적화 배치',r.optimization_batch+'쌍'],['조건당 반복',r.repetitions+'회'],['Target 추출',r.replacement?'복원 추출':'비복원 추출'],['반복 집계',r.aggregation==='mean_log_gamma'?'최적 log γ의 평균':r.aggregation],['log γ 탐색',`${r.log_gamma_min} ~ ${r.log_gamma_max} · ${r.log_gamma_points}개 후보`],['Adams log shape',r.adams_log_shape],['손실 정밀도',r.loss_precision],['모델 가중치','고정']])}</div></div>`;
 out+=`<details class="panel detail-panel" id="storage-details"><summary>저장 위치와 모델 식별</summary>${kv([['비교 결과',p.output],['RBF 결과',r?.output],['모델 가중치',r?.checkpoint],['VAE',r?.vae],['가중치 SHA256',r?.checkpoint_sha256],['기준 코드 커밋',r?.source_commit],['준비 검증 기록',r?.preparation_validation]])}</details><details class="panel detail-panel" id="source-details"><summary>조건의 근거와 원본 기록</summary><p class="footnote">아래는 저장된 근거 기록입니다. 초기 준비 시점의 설명도 보존하며, 실제 실행 상태와 측정 배치는 현재 결과 기록을 기준으로 표시합니다.</p><h3 style="margin-top:18px">샘플러 비교</h3><div class="source-list">${sourceList(p.sources)}</div>${r?`<h3 style="margin-top:24px">RBF 파이프라인</h3><div class="source-list">${sourceList(r.sources)}</div>`:''}</details><details class="panel detail-panel" id="raw-settings"><summary>전체 설정 원문</summary><pre>${esc(JSON.stringify({comparison:p,rbf:r||null,fid_validation:f},null,2))}</pre></details>`;
 html('settings-content',out);open.forEach(id=>{if($(id))$(id).open=true});settingsKey=key;
}
function renderCoefficients(){
 text('coefficient-description',`논문 Figure 5 형식 · SiT-S/2 CFG ${rbfCFG}의 실제 학습 결과`);
 if(!coeffs?.ready||coeffs.cfg!==rbfCFG){$('coefficient-figure').className='empty';html('coefficient-figure',coeffs?.cfg===rbfCFG?'해당 CFG의 학습과 계수 검증이 완료되면 표시됩니다.':'선택한 CFG의 계수를 확인하고 있습니다.');html('coefficient-table','');return;}
 const options=['all',...coeffs.conditions.map(r=>String(r.nfe))];
 if($('coefficient-controls').dataset.options!==JSON.stringify(options)){
  if(!options.includes(coefficientNFE))coefficientNFE='all';
  $('coefficient-controls').dataset.options=JSON.stringify(options);
  html('coefficient-controls',options.map(n=>`<button data-coefficient-nfe="${n}" aria-pressed="${n===coefficientNFE}">${n==='all'?'전체':'NFE '+n}</button>`).join(''));
  $('coefficient-controls').querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{coefficientNFE=b.dataset.coefficientNfe;renderCoefficients()}));
 }
 $('coefficient-controls').querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',b.dataset.coefficientNfe===coefficientNFE));
 const part=coefficientNFE==='all'?'all':'nfe'+coefficientNFE,url='/rbf-coefficients/'+rbfCFG+'/coefficients_'+part+'.svg?v='+coeffs.updated_at;
 $('coefficient-figure').className='coefficient-figure';
 html('coefficient-figure',`<button class="preview-button" data-preview="${url}" data-label="CFG ${rbfCFG} · 학습 파라미터와 CMR · ${coefficientNFE==='all'?'NFE 전체':'NFE '+coefficientNFE}" aria-label="학습 계수 그래프 확대"><img style="max-height:none;background:white" src="${url}" alt="논문 Figure 5 방식: log gamma, predictor CMR, corrector CMR 3행 그래프"></button>`);
 const selected=coeffs.conditions.filter(r=>coefficientNFE==='all'||String(r.nfe)===coefficientNFE);
 const vector=v=>v===null?'미실행':v.map(c=>(c>=0?'+':'')+num(c,6)).join(', ');
 html('coefficient-table',table(['NFE','Step','Δt','log γ pred','log γ corr','Predictor cⱼ','Corrector cⱼ'],selected.flatMap(r=>r.steps.map((i)=>`<tr><th scope="row">${r.nfe}</th><td>${i}</td><td>${num(r.timesteps[i+1]-r.timesteps[i],6)}</td><td>${r.predictor_log_gamma[i]===null?'미사용':num(r.predictor_log_gamma[i],4)}</td><td>${r.corrector_log_gamma[i]===null?'미실행':num(r.corrector_log_gamma[i],4)}</td><td>${vector(r.predictor_coefficients[i])}</td><td>${vector(r.corrector_coefficients[i])}</td></tr>`))));
}
function showConnection(){
 const failed=Object.entries(errors).filter(([,e])=>e),data={compare,rbf,coefficients:coeffs},labels={compare:'비교·GPU',rbf:'RBF',coefficients:'학습 계수'};
 $('connection-dot').className='dot '+(failed.length?'offline':compare||rbf?'online':'');
 const timestamps=[compare?.updated_at,rbf?.updated_at].filter(finite),last=timestamps.length?Math.min(...timestamps):null;
 text('connection-text',failed.length?'일부 상태 갱신 실패':last?`${timeText(last)} 갱신`:'상태 불러오는 중');
 $('connection-alert').hidden=!failed.length;
 if(failed.length)text('connection-alert',failed.map(([k,e])=>`${labels[k]} 상태를 갱신하지 못했습니다. ${data[k]?`마지막 확인 ${timeText(data[k].updated_at)}의 기록을 표시합니다.`:'확인된 기록이 없습니다.'} (${e})`).join(' '));
}
async function fetchData(url){const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),9000);try{const r=await fetch(url,{cache:'no-store',signal:controller.signal});if(!r.ok)throw new Error('HTTP '+r.status);return await r.json()}finally{clearTimeout(timer)}}
async function refresh(){
 if(busy)return;busy=true;$('refresh').disabled=true;
 const requestedCFG=rbfCFG;
 try{await Promise.all([fetchData('/api/rbf-coefficients?cfg='+requestedCFG).then(d=>{if(requestedCFG!==rbfCFG)return;coeffs={...d,cfg:requestedCFG};errors.coefficients=null;renderCoefficients()}).catch(e=>{errors.coefficients=e.message}),fetchData('/api/comparison').then(d=>{if(!Array.isArray(d.items)||!d.protocol)throw new Error('응답 형식 오류');compare=d;errors.compare=null;renderCompare()}).catch(e=>{errors.compare=e.name==='AbortError'?'응답 시간 초과':e.message}),fetchData('/api/rbf-training').then(d=>{if(!Array.isArray(d.targets)||!d.protocol)throw new Error('응답 형식 오류');rbf=d;errors.rbf=null;renderRBF()}).catch(e=>{errors.rbf=e.name==='AbortError'?'응답 시간 초과':e.message})]);}finally{busy=false;$('refresh').disabled=false;showConnection()}
}
document.addEventListener('click',e=>{const b=e.target.closest('[data-preview]');if(!b)return;$('large-preview').src=b.dataset.preview;$('large-preview').alt=b.dataset.label;$('preview-dialog').classList.toggle('wide',b.dataset.preview.includes('/rbf-coefficients/'));text('preview-title',b.dataset.label);$('preview-dialog').showModal()});
$('previews').addEventListener('error',e=>{if(e.target.tagName==='IMG'){const button=e.target.closest('button');if(button)button.outerHTML='<div class="preview-placeholder">미리보기 이미지를 불러오지 못했습니다.</div>'}},true);
$('close-preview').addEventListener('click',()=>$('preview-dialog').close());
$('refresh').addEventListener('click',refresh);
refresh();setInterval(refresh,3000);
})();
