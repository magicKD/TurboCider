"""Precision SVG schematics; all arrows encode actual data dependencies."""
from pathlib import Path
from html import escape
P=Path(__file__).resolve().parents[1]/'notes/figures';P.mkdir(parents=True,exist_ok=True)
a=['<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="1010" viewBox="0 0 1440 1010"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10z" fill="#475569"/></marker></defs><rect width="1440" height="1010" fill="white"/>']
def text(x,y,s,size=20,color='#334155'):a.append(f'<text x="{x}" y="{y}" fill="{color}" font-family="Arial,sans-serif" font-size="{size}">{escape(s)}</text>')
def box(x,y,w,h,lines,kind='gpu'):
 fill,stroke={'gpu':('#dbeafe','#1e40af'),'ane':('#dcfce7','#166534'),'cpu':('#f3e8ff','#6b21a8'),'neutral':('#f1f5f9','#475569')}[kind]
 a.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
 for i,s in enumerate(lines):text(x+14,y+29+i*25,s,18)
def arrow(x,y,xx,yy):a.append(f'<path d="M{x} {y} L{xx} {yy}" fill="none" stroke="#475569" stroke-width="2" marker-end="url(#arrow)"/>')
text(35,42,'Transformer heterogeneity: legal partitions and synchronization',29,'#1e293b')
text(35,76,'Batch 1 • causal attention • independent weights per block • FP16 activations',19)
text(35,122,'A. Sequential pre-norm block: the FFN depends on the attention residual',23)
box(35,147,160,70,['X [S,H]','GPU RMSNorm']);arrow(195,182,230,182)
box(230,147,255,70,['GPU QKV → attention','output projection']);arrow(485,182,520,182)
box(520,147,230,70,['Add residual','GPU RMSNorm']);arrow(750,182,785,182)
box(785,147,245,70,['Parallel FFN shards','details below'],'ane');arrow(1030,182,1065,182)
box(1065,147,335,70,['GPU join + residual → Y','Y feeds the next independent block'])
text(35,268,'B. Tensor parallel FFN: split paired intermediate channels, reduce hidden-size partials',23)
box(35,330,190,80,['Normalized X','[S,H]'],'neutral')
for y,kind,lines in [(290,'gpu',['GPU: up/gate prefix','SwiGLU → down prefix']),(402,'ane',['ANE: up/gate suffix','SwiGLU → down suffix'])]:
 box(330,y,360,80,lines,kind);arrow(225,370,330,y+40);arrow(690,y+40,865,370)
box(865,330,430,80,['GPU: P_gpu + P_ane + residual','Each partial is [S,H]; one join'])
text(330,521,'CPU in the three-device ablation computes a small up/gate shard;',17,'#6b21a8')
text(330,546,'its activation and down projection are consumed on GPU.',17,'#6b21a8')
text(35,593,'C. Sequence-row FFN: each branch has full weights, disjoint token rows',23)
box(35,625,240,78,['GPU: X[0:r,:]','complete FFN prefix'])
box(375,625,280,78,['ANE: X[r:S,:]','complete FFN suffix'],'ane')
arrow(275,664,300,728);arrow(300,728,760,728);arrow(655,664,760,716)
box(775,674,575,76,['GPU: row-wise assembly + residual','Attention still uses all causal keys; no local-only attention'])
text(35,798,'D. Head parallel attention: each assigned head sees the complete causal sequence',23)
box(35,825,245,78,['GPU QKV, all heads','shared Q/K/V [S,3H]']);arrow(280,864,330,864)
box(330,825,350,78,['GPU prefix heads: QK →','causal softmax → PV'])
box(750,825,380,78,['ANE or CPU suffix heads:','QK → causal softmax → PV'],'ane')
arrow(280,864,305,941);arrow(305,941,940,941);arrow(940,941,940,903)
arrow(680,864,715,916);arrow(715,916,1180,928);arrow(1130,864,1180,928)
text(1070,963,'Join → GPU W_o',20)
a.append('</svg>');(P/'architecture.svg').write_text(''.join(a))
