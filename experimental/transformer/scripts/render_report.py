"""Render this report's deliberately simple Markdown as standalone local HTML."""
import re,html
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'notes'
lines=(P/'REPORT.md').read_text().splitlines();out=[];toc=[];i=0

def inline(s):
    s=html.escape(s)
    s=re.sub(r'!\[([^]]*)\]\(([^)]+)\)',r'<img alt="\1" src="\2" loading="lazy">',s)
    s=re.sub(r'(?<!!)\[([^]]*)\]\(([^)]+)\)',r'<a href="\2">\1</a>',s)
    s=re.sub(r'`([^`]+)`',r'<code>\1</code>',s)
    s=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',s)
    s=re.sub(r'\*([^*]+)\*',r'<em>\1</em>',s)
    return s
while i<len(lines):
    line=lines[i]
    if not line.strip():i+=1;continue
    if line.startswith('```'):
        code=[];i+=1
        while i<len(lines) and not lines[i].startswith('```'):code.append(lines[i]);i+=1
        out.append('<pre><code>'+html.escape('\n'.join(code))+'</code></pre>');i+=1;continue
    if line.startswith('#'):
        level=len(line)-len(line.lstrip('#'));title=line[level:].strip();ident=f'heading-{i}'
        if level==2:toc.append((ident,title))
        out.append(f'<h{level} id="{ident}">{inline(title)}</h{level}>');i+=1;continue
    if line.startswith('|'):
        trs=[];j=0
        while i<len(lines) and lines[i].startswith('|'):
            row=lines[i];i+=1
            if re.fullmatch(r'[| :\-]+',row):continue
            tag='th' if j==0 else 'td';trs.append('<tr>'+''.join(f'<{tag}>{inline(x.strip())}</{tag}>' for x in row.strip('|').split('|'))+'</tr>');j+=1
        out.append('<div class="table-wrap"><table>'+''.join(trs)+'</table></div>');continue
    if re.match(r'\d+\. ',line):
        lis=[]
        while i<len(lines) and re.match(r'\d+\. ',lines[i]):lis.append('<li>'+inline(re.sub(r'^\d+\. ','',lines[i]))+'</li>');i+=1
        out.append('<ol>'+''.join(lis)+'</ol>');continue
    para=[line];i+=1
    while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','```')):para.append(lines[i]);i+=1
    out.append('<p>'+inline(' '.join(para))+'</p>')
style='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f5f6f4;color:#202a30;font-family:Georgia,"Songti SC",serif;font-size:17px;line-height:1.85}main{max-width:1100px;margin:36px auto;padding:54px 64px;background:white;border-top:5px solid #166534;box-shadow:0 3px 24px #0000000a}h1,h2,h3,nav,th{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}h1{font-size:35px;line-height:1.4;letter-spacing:-.7px}h2{font-size:26px;margin-top:52px;padding-top:18px;border-top:1px solid #d9e2dc}h3{font-size:20px;margin-top:30px}p{margin:16px 0}a{color:#166534;text-underline-offset:3px}img{display:block;max-width:100%;height:auto;margin:24px auto 8px}em{font-size:14px;color:#54626b}code{font-family:Menlo,monospace;font-size:.83em;background:#f1f5f4;padding:2px 4px}pre{overflow:auto;background:#f1f5f4;border-left:3px solid #94b8a1;padding:18px;line-height:1.6}pre code{padding:0}.table-wrap{overflow-x:auto;margin:24px 0}table{width:100%;border-collapse:collapse;font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;font-size:13px;line-height:1.6}th{background:#eaf2ec;text-align:left;border-bottom:2px solid #49785b}th,td{padding:10px 9px;vertical-align:top;border-bottom:1px solid #e0e6e2}tr:nth-child(even){background:#fafcfb}nav{padding:20px;background:#f6f8f6;font-size:14px;line-height:2}nav a{display:inline-block;margin-right:18px}footer{margin-top:40px;font-size:13px;color:#64748b}@media(max-width:760px){main{margin:0;padding:24px 20px}h1{font-size:27px}body{font-size:16px}}@media print{body{background:white}main{margin:0;padding:0;box-shadow:none;border:0}nav{display:none}h2,h3{break-after:avoid}img,table{break-inside:avoid}a{color:inherit}body{font-size:11pt}.table-wrap{overflow:visible}}'''
nav='<nav aria-label="目录">'+''.join(f'<a href="#{ident}">{html.escape(t)}</a>' for ident,t in toc)+'</nav>'
body=out[0]+nav+''.join(out[1:])
(P/'REPORT.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TurboCider Transformer 异构并行技术报告</title><style>'+style+'</style><main>'+body+'<footer>Local, reproducible research artifact · SVG figures and raw measurements retained · TurboCider 2026</footer></main></html>')
