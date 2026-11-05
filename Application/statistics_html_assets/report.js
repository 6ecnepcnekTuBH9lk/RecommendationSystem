'use strict';
const specs=JSON.parse(document.getElementById('chart-data').textContent);
const colors=['#a65cf2','#207e91','#8c9cab','#b3a2df'];
window.reportErrors=[];
function selectedSpec(node) {
  const index=node.dataset.chart,base=specs[Number(index)];
  if(!base.variants)return base;
  const windowSelect=document.querySelector('[data-rfm-window="'+index+'"]');
  const currency=document.querySelector('[data-rfm-currency="'+index+'"]');
  return base.variants[windowSelect.value+'|'+currency.value];
}
async function drawChart(node,force=false) {
  if(node.closest('[hidden]')||node.dataset.ready&&!force)return;
  const spec=selectedSpec(node);
  if(!spec.data.length) {
    if(node.classList.contains('js-plotly-plot'))Plotly.purge(node);
    node.textContent='Нет данных для диаграммы.';node.dataset.ready='empty';return;
  }
  if(!node.classList.contains('js-plotly-plot'))node.textContent='';
  const data=spec.data.map((trace,index)=>{
    const copy={...trace};delete copy.exact;
    const coordinate=copy.type==='pie'?'values':spec.horizontal?'x':'y';
    copy[coordinate]=copy[coordinate].map(v=>v===null?null:Number(v));
    if(copy.type!=='pie')copy.marker={color:colors[spec.secondary&&index===0?1:index%colors.length]};
    else copy.marker={colors};
    return copy;
  });
  const legend=data.length>1||data[0].type==='pie';
  const layout={autosize:true,separators:', ',barmode:'group',
    margin:{t:12,r:24,b:legend?105:65,l:68},paper_bgcolor:'#fff',plot_bgcolor:'#fff',
    colorway:spec.secondary?[colors[1],colors[0]]:colors,
    font:{family:'Arial, sans-serif',size:12,color:'#465766'},showlegend:legend,
    legend:{orientation:'h',y:-.32,x:0},hoverlabel:{align:'left',font:{size:12}},
    xaxis:{gridcolor:'#edf1f5',zeroline:false,automargin:true},
    yaxis:{gridcolor:'#edf1f5',zeroline:false,rangemode:'tozero',automargin:true}};
  if(spec.horizontal){
    layout.yaxis={type:'category',autorange:'reversed',tickvals:spec.labels.map((_,i)=>String(i)),
      ticktext:spec.labels,automargin:true};
    layout.margin.l=120;
  } else layout.xaxis.type='category';
  if(data[0].type!=='pie'){
    const axis=spec.horizontal?layout.xaxis:layout.yaxis;
    const values=data.flatMap(t=>t[spec.horizontal?'x':'y']).filter(v=>v!==null);
    const maximum=Math.max(0,...values);
    axis.tickformat=maximum>=1000?'.3~s':spec.money?',.2~f':',d';
    if(spec.percentage){axis.range=[0,100];axis.dtick=25;axis.title={text:'Повторная доля, %'};}
    else if(!spec.money&&maximum<=5)axis.dtick=1;
    if(spec.zero_is_data&&!spec.horizontal){
      layout.annotations=spec.data[0].exact.flatMap((value,i)=>value===null||Number(value)===0?
        [{x:spec.labels[i],y:0,text:value===null?'—':spec.percentage?'0 %':'0',showarrow:false,yshift:12}]:[]);
    }
  }
  if(spec.horizontal&&spec.labels.length>12){
    node.style.height=Math.max(340,spec.labels.length*24+90)+'px';
    node.parentElement.style.maxHeight='650px';node.parentElement.style.overflowY='auto';
  }
  node.dataset.ready='pending';
  try {
    await Plotly.react(node,data,layout,{responsive:true,displaylogo:false,showLink:false,
      typesetMath:false,showTips:false,scrollZoom:false,modeBarButtonsToRemove:['sendDataToCloud','lasso2d','select2d']});
    node.dataset.ready='true';
  } catch(error) {
    window.reportErrors.push(String(error));node.dataset.ready='error';node.textContent='Не удалось отобразить диаграмму.';
  }
}
async function drawSection(panel) {
  await Promise.all([...panel.querySelectorAll('[data-chart]')].map(node=>drawChart(node)));
  for(const node of panel.querySelectorAll('.js-plotly-plot'))if(!node.closest('[hidden]'))await Plotly.Plots.resize(node);
}
const tabs=[...document.querySelectorAll('[role=tab]')];
function activate(tab,focus=false) {
  const siblings=[...tab.closest('[role=tablist]').querySelectorAll(':scope > [role=tab]')];
  siblings.forEach(button=>{
    const active=button===tab;button.setAttribute('aria-selected',String(active));button.tabIndex=active?0:-1;
    document.getElementById(button.getAttribute('aria-controls')).hidden=!active;
  });
  if(focus)tab.focus();
  window.reportReady=drawSection(document.getElementById(tab.getAttribute('aria-controls')));
}
tabs.forEach(tab=>{
  tab.addEventListener('click',()=>activate(tab));
  tab.addEventListener('keydown',event=>{
    const siblings=[...tab.closest('[role=tablist]').querySelectorAll(':scope > [role=tab]')];
    const index=siblings.indexOf(tab);let target=index;
    if(event.key==='ArrowRight')target=(index+1)%siblings.length;
    else if(event.key==='ArrowLeft')target=(index+siblings.length-1)%siblings.length;
    else if(event.key==='Home')target=0;
    else if(event.key==='End')target=siblings.length-1;
    else return;
    event.preventDefault();activate(siblings[target],true);
  });
});
for(const select of document.querySelectorAll('[data-rfm-window],[data-rfm-currency]')) {
  select.addEventListener('change',()=>{
    const index=select.dataset.rfmWindow||select.dataset.rfmCurrency;
    window.reportReady=drawChart(document.getElementById('chart-'+index),true);
  });
}
function decimalCompare(left,right) {
  const parse=value=>{
    const [whole,fraction='']=value.split('.');
    return [BigInt(whole.replace('-','')+fraction)*(whole.startsWith('-')?-1n:1n),fraction.length];
  };
  const [a,as]=parse(left),[b,bs]=parse(right);
  const aa=a*10n**BigInt(Math.max(0,bs-as)),bb=b*10n**BigInt(Math.max(0,as-bs));
  return aa<bb?-1:aa>bb?1:0;
}
document.querySelectorAll('table').forEach(table=>{
  table.querySelectorAll('th button').forEach(button=>button.addEventListener('click',()=>{
    const header=button.closest('th'),column=Number(button.dataset.column);
    const descending=header.getAttribute('aria-sort')==='ascending';
    table.querySelectorAll('th').forEach(th=>th.setAttribute('aria-sort','none'));
    header.setAttribute('aria-sort',descending?'descending':'ascending');
    const rows=[...table.tBodies[0].querySelectorAll('tr[data-order]')];
    rows.sort((left,right)=>{
      const a=left.cells[column],b=right.cells[column];let comparison;
      if(a.dataset.kind==='empty'||b.dataset.kind==='empty')comparison=(a.dataset.kind==='empty')-(b.dataset.kind==='empty');
      else if(a.dataset.kind==='number'&&b.dataset.kind==='number')comparison=decimalCompare(a.dataset.sort,b.dataset.sort);
      else comparison=a.dataset.sort.localeCompare(b.dataset.sort,'ru');
      return(descending?-comparison:comparison)||Number(left.dataset.order)-Number(right.dataset.order);
    });
    table.tBodies[0].append(...rows);
  }));
});
document.querySelectorAll('input[data-table]').forEach(input=>input.addEventListener('input',()=>{
  const table=document.getElementById(input.dataset.table),term=input.value.toLocaleLowerCase('ru');
  table.querySelectorAll('tbody tr[data-order]').forEach(row=>{row.hidden=!row.textContent.toLocaleLowerCase('ru').includes(term);});
}));
function sizeNavigation(){
  const nav=document.querySelector('body > nav');
  document.documentElement.style.setProperty('--main-nav-height',nav.getBoundingClientRect().height+'px');
}
sizeNavigation();
activate(tabs[0]);
let resizeTimer;
window.addEventListener('resize',()=>{
  sizeNavigation();
  clearTimeout(resizeTimer);
  resizeTimer=setTimeout(()=>{
    const panel=document.querySelector('main > [role=tabpanel]:not([hidden])');
    window.reportReady=drawSection(panel);
  },100);
});
