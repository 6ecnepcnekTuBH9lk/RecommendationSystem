'use strict';
const specs = JSON.parse(document.getElementById('chart-data').textContent);
const colors = ['#367ca5','#227d91','#8c9cab','#a6c4d8'];
window.reportErrors = [];
async function drawSection(section) {
  const jobs = [];
  section.querySelectorAll('[data-chart]').forEach(node => {
    const spec = specs[Number(node.dataset.chart)];
    if (node.dataset.ready || !spec.data.length) return;
    node.dataset.ready = 'pending';
    const data = spec.data.map(trace => {
      const copy = {...trace};
      delete copy.exact;
      // Only coordinates use native numbers; display formatting never feeds back into these values.
      const coordinate = copy.type==='pie'?'values':spec.horizontal?'x':'y';
      copy[coordinate] = copy[coordinate].map(value => Number(value));
      return copy;
    });
    const layout = {autosize:true,separators:', ',margin:{t:12,r:20,b:66,l:68},paper_bgcolor:'#fff',plot_bgcolor:'#fff',
      colorway:colors,font:{family:'Arial, sans-serif',size:12,color:'#465766'},showlegend:true,
      legend:{orientation:'h',y:-.22,x:0},hoverlabel:{align:'left',font:{size:12}},
      xaxis:{gridcolor:'#edf1f5',zeroline:false,tickformat:spec.monthly?'%m.%Y':'~s',automargin:true},
      yaxis:{gridcolor:'#edf1f5',zeroline:false,rangemode:'tozero',tickformat:'~s',automargin:true}};
    if (spec.monthly) { layout.xaxis.type='date'; layout.dragmode='zoom'; }
    else if (!spec.horizontal) layout.xaxis.type='category';
    if (spec.horizontal) {
      const shortLabels=spec.labels.map(label=>node.clientWidth<480&&label.length>18?label.slice(0,17)+'…':label);
      layout.yaxis={type:'category',autorange:'reversed',tickvals:spec.labels.map((_,i)=>String(i)),
        ticktext:shortLabels,automargin:true};
      layout.margin.l=node.clientWidth<480?100:120;
    }
    if(data[0].type!=='pie') {
      const axis=spec.horizontal?layout.xaxis:layout.yaxis;
      const maximum=Math.max(...data.flatMap(trace=>trace[spec.horizontal?'x':'y']));
      axis.tickformat=maximum>=1000?'.3~s':spec.money?',.2~f':',d';
      if(!spec.money&&maximum<=5)axis.dtick=1;
    }
    const job = Plotly.newPlot(node,data,layout,{responsive:true,displaylogo:false,showLink:false,
      typesetMath:false,showTips:false,scrollZoom:false,modeBarButtonsToRemove:['sendDataToCloud','lasso2d','select2d']})
      .then(()=>{node.dataset.ready='true';})
      .catch(error=>{window.reportErrors.push(String(error));node.textContent='Не удалось отобразить диаграмму.';});
    jobs.push(job);
  });
  await Promise.all(jobs);
  for(const node of section.querySelectorAll('.js-plotly-plot')) {
    const spec=specs[Number(node.dataset.chart)];
    if(spec.horizontal)await Plotly.relayout(node,{'yaxis.ticktext':spec.labels.map(label=>
      node.clientWidth<480&&label.length>18?label.slice(0,17)+'…':label)});
    await Plotly.Plots.resize(node);
  }
}
const tabs = [...document.querySelectorAll('[role=tab]')];
function activate(tab, focus=false) {
  tabs.forEach(button=>{
    const active=button===tab;
    button.setAttribute('aria-selected',String(active));button.tabIndex=active?0:-1;
    document.getElementById(button.getAttribute('aria-controls')).hidden=!active;
  });
  if(focus) tab.focus();
  window.reportReady=drawSection(document.getElementById(tab.getAttribute('aria-controls')));
}
tabs.forEach((tab,index)=>{
  tab.addEventListener('click',()=>activate(tab));
  tab.addEventListener('keydown',event=>{
    let target=index;
    if(event.key==='ArrowRight') target=(index+1)%tabs.length;
    else if(event.key==='ArrowLeft') target=(index+tabs.length-1)%tabs.length;
    else if(event.key==='Home') target=0;
    else if(event.key==='End') target=tabs.length-1;
    else return;
    event.preventDefault();activate(tabs[target],true);
  });
});
// Exact decimal comparison: no Number conversion when sorting financial values.
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
      const a=left.cells[column],b=right.cells[column];
      let comparison;
      if(a.dataset.kind==='empty'||b.dataset.kind==='empty') comparison=(a.dataset.kind==='empty')-(b.dataset.kind==='empty');
      else if(a.dataset.kind==='number'&&b.dataset.kind==='number') comparison=decimalCompare(a.dataset.sort,b.dataset.sort);
      else comparison=a.dataset.sort.localeCompare(b.dataset.sort,'ru');
      return (descending?-comparison:comparison)||Number(left.dataset.order)-Number(right.dataset.order);
    });
    table.tBodies[0].append(...rows);
  }));
});
document.querySelectorAll('input[data-table]').forEach(input=>input.addEventListener('input',()=>{
  const table=document.getElementById(input.dataset.table),term=input.value.toLocaleLowerCase('ru');
  table.querySelectorAll('tbody tr[data-order]').forEach(row=>{
    row.hidden=!row.textContent.toLocaleLowerCase('ru').includes(term);
  });
}));
activate(tabs[0]);
let resizeTimer;
window.addEventListener('resize',()=>{
  clearTimeout(resizeTimer);
  resizeTimer=setTimeout(()=>{
    const panel=document.querySelector('[role=tabpanel]:not([hidden])');
    window.reportReady=drawSection(panel);
  },100);
});
