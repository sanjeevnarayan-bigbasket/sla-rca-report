import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import io
import streamlit.components.v1 as components

st.set_page_config(
    page_title="SLA Breach — RCA Report",
    page_icon="📊",
    layout="wide"
)

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════
SLA_CONFIG = {
    'InhouseSLA':   {'sla':180,'buffer':30, 'hard_limit':210,'avoidable':False},
    'OP to INP':    {'sla':20, 'buffer':5,  'hard_limit':25, 'avoidable':False},
    'INP to PK':    {'sla':120,'buffer':30, 'hard_limit':150,'avoidable':False},
    'PK to BIN':    {'sla':20, 'buffer':5,  'hard_limit':25, 'avoidable':False},
    'BIN to RTS':   {'sla':60, 'buffer':30, 'hard_limit':90, 'avoidable':False},
    'RTS to Reach': {'sla':600,'buffer':300,'hard_limit':900,'avoidable':True},
    'RTS to DEL':   {'sla':600,'buffer':300,'hard_limit':900,'avoidable':True},
}
DUMP_COL_MAP = {
    'Op to Inp 1':      'OP to INP',
    'Inp to Pk 1':      'INP to PK',
    'Pk to Bin 1':      'PK to BIN',
    'Bin to Rts 1':     'BIN to RTS',
    'RTS TO Reached 1': 'RTS to Reach',
    'Rts to del 1':     'RTS to DEL',
}
STAGE_COLS       = list(DUMP_COL_MAP.values())
INDIVIDUAL_SECS  = [f'{s}_secs' for s in STAGE_COLS]
BREAKDOWN_STAGES = ['OP to INP','INP to PK','PK to BIN','InhouseSLA','BIN to RTS']
SUMMARY_STAGES   = ['OP to INP','INP to PK','PK to BIN','InhouseSLA','BIN to RTS']
TOTAL_COLS       = 9

# ═══════════════════════════════════════════════════════════════
#  REGION / TIER MAPPING
# ═══════════════════════════════════════════════════════════════
T1_REGIONS = ['Ahmedabad-Gandhinagar','Bangalore','Chennai','Gurgaon',
              'Hyderabad','Kolkata','Mumbai','Noida','Pune']
T2_REGIONS = ['Agra','Allahabad','Bhopal','Bhubaneshwar-Cuttack',
              'Chandigarh Tricity','DehraDun','Guwahati','Indore',
              'Kochi','Kozhikode','Lucknow-Kanpur','Mangaluru','Mysore',
              'Nagpur','Patna','Raipur','Ranchi','Surat',
              'Thiruvananthapuram','Vadodara','Vijayawada-Guntur','Visakhapatnam']
T3_REGIONS = ['Bangalore Rural','Chennai Rural','Hyderabad Rural',
              'Kolkata Rural','Noida Rural','Ranchi Rural','Vizag Rural']

CITY_TO_REGION = {
    'BLR':'Bangalore',    'HYD':'Hyderabad',   'KOL':'Kolkata',
    'GUR':'Gurgaon',      'NOI':'Noida',        'MUM':'Mumbai',
    'CHN':'Chennai',      'AHD':'Ahmedabad-Gandhinagar','PUN':'Pune',
    'AGA':'Agra',         'ALH':'Allahabad',    'BPL':'Bhopal',
    'BHU':'Bhubaneshwar-Cuttack','MOH':'Chandigarh Tricity',
    'DDN':'DehraDun',     'GUW':'Guwahati',     'IND':'Indore',
    'KOC':'Kochi',        'KZK':'Kozhikode',    'LUC':'Lucknow-Kanpur',
    'MGL':'Mangaluru',    'MYS':'Mysore',       'NAG':'Nagpur',
    'PAT':'Patna',        'RAI':'Raipur',       'RNC':'Ranchi',
    'SRT':'Surat',        'TVP':'Thiruvananthapuram','VAD':'Vadodara',
    'VJY':'Vijayawada-Guntur','VIZ':'Visakhapatnam','DEL':'Gurgaon',
}
T3_CITY_TO_REGION = {
    'BLR':'Bangalore Rural','KOL':'Kolkata Rural','HYD':'Hyderabad Rural',
    'NOI':'Noida Rural','RNC':'Ranchi Rural','VZG':'Vizag Rural',
    'CHN':'Chennai Rural',
}

def get_tier(r):
    if r in T1_REGIONS:   return 'T1'
    elif r in T2_REGIONS: return 'T2'
    elif r in T3_REGIONS: return 'T3'
    return 'Unknown'

def extract_region(store_name):
    try:
        parts = str(store_name).strip().split('-')
        if len(parts) < 2: return 'Unknown'
        code = parts[1]; tp = code[:2]; cc = code[2:]
        if tp == 'T3': return T3_CITY_TO_REGION.get(cc,'Unknown')
        return CITY_TO_REGION.get(cc,'Unknown')
    except: return 'Unknown'

# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════
def time_to_seconds(val):
    try:
        if val is None: return 0
        if isinstance(val,(int,float)):
            if np.isnan(val): return 0
            if 0 < val < 1: return int(round(val*86400))
            return 0
        if isinstance(val,str):
            v = val.strip()
            if v in ['','-','nan','0:00:00','00:00:00']: return 0
            p = v.split(':')
            if len(p)==3:
                return int(p[0])*3600+int(p[1])*60+int(float(p[2]))
            return 0
        if hasattr(val,'total_seconds'): return max(0,int(val.total_seconds()))
        if hasattr(val,'hour'): return val.hour*3600+val.minute*60+val.second
        return 0
    except: return 0

def to_readable(secs):
    if not secs or secs<=0: return "0s"
    s=int(secs)
    if s>=3600: return f"{s//3600}h {(s%3600)//60}m {s%60}s"
    elif s>=60: return f"{s//60}m {s%60}s"
    return f"{s}s"

def severity_label(actual, hard_limit):
    if actual<=0: return "OK"
    pct=((actual-hard_limit)/hard_limit)*100
    if pct>=100: return "CRITICAL"
    elif pct>=50: return "HIGH"
    return "BREACH"

def sev_color(sev):
    return {"CRITICAL":"#dc2626","HIGH":"#ea580c",
            "BREACH":"#d97706","OK":"#16a34a"}.get(sev,"#64748b")

def get_priority(row_data):
    ih=bool(row_data.get('InhouseSLA_breach',False))
    br=bool(row_data.get('BIN to RTS_breach',False))
    if ih and br: return "HIGH"
    elif ih or br: return "MEDIUM"
    return "OK"

def priority_sort_key(p):
    return {"HIGH":0,"MEDIUM":1,"OK":2}.get(p,3)

def metric_color(pct):
    if pct>=70: return "#16a34a"
    elif pct>=50: return "#ea580c"
    return "#dc2626"

# ═══════════════════════════════════════════════════════════════
#  OTR / MOD NORMALIZATION
# ═══════════════════════════════════════════════════════════════
def normalize_otr_val(val):
    try:
        if val is None or (isinstance(val,float) and np.isnan(val)): return ''
        if isinstance(val,bool): return '1' if val else '0'
        if isinstance(val,(int,float)):
            iv=int(val); return str(iv)
        s=str(val).strip()
        try:
            f=float(s)
            if f==int(f): return str(int(f))
            return s
        except: return s
    except: return str(val).strip()

def normalize_mod_val(val):
    try:
        if val is None: return '0'
        if isinstance(val, bool): return '1' if val else '0'
        if isinstance(val, float):
            if np.isnan(val): return '0'
            return '1' if int(val)==1 else '0'
        if isinstance(val, int): return '1' if val==1 else '0'
        s=str(val).strip()
        if not s or s.lower() in ('nan','none',''): return '0'
        try: return '1' if int(float(s))==1 else '0'
        except: return '0'
    except: return '0'

def compute_otr_pct(all_del_df, tier=None, region=None, store=None):
    ds=all_del_df
    if tier   is not None: ds=ds[ds['Tier']  ==tier]
    if region is not None: ds=ds[ds['Region']==region]
    if store  is not None: ds=ds[ds['Store'] ==store]
    total=len(ds)
    if total==0: return 0
    return int((ds['OTR']=='10').sum()/total*100)

def compute_mod_pct(all_del_df, tier=None, region=None, store=None):
    ds=all_del_df
    if tier   is not None: ds=ds[ds['Tier']  ==tier]
    if region is not None: ds=ds[ds['Region']==region]
    if store  is not None: ds=ds[ds['Store'] ==store]
    total=len(ds)
    if total==0: return 0
    return int((ds['MOD']=='1').sum()/total*100)

def compute_otr_store_brackets(all_del_df, tier=None, region=None):
    ds=all_del_df
    if tier   is not None: ds=ds[ds['Tier']  ==tier]
    if region is not None: ds=ds[ds['Region']==region]
    if ds.empty: return 0,0,0
    above50=0; below_eq50=0
    for store in ds['Store'].unique():
        s_ds=ds[ds['Store']==store]
        t=len(s_ds)
        if t==0: continue
        pct=int((s_ds['OTR']=='10').sum()/t*100)
        if pct>50: above50+=1
        else: below_eq50+=1
    return above50+below_eq50, above50, below_eq50

# ═══════════════════════════════════════════════════════════════
#  SPLIT BAR
# ═══════════════════════════════════════════════════════════════
def split_bar(wp, bp, wn=0, bn=0, width=110):
    if wp==0 and bp==0:
        return "<span style='color:#94a3b8;font-size:10px;'>—</span>"
    gp=max(0.0,min(100.0,wp)); rp=max(0.0,min(100.0,bp))
    def pct_lbl(pct): return f"{pct:.0f}%" if pct>=5 else ""
    gmin="2px" if gp>0 else "0"; rmin="2px" if rp>0 else "0"
    return f"""<div style='display:inline-flex;width:{width}px;height:18px;
                border-radius:3px;overflow:hidden;vertical-align:middle;
                border:1px solid #e2e8f0;'>
      <div style='width:{gp:.1f}%;min-width:{gmin};background:#16a34a;
                  display:flex;align-items:center;justify-content:center;'>
        <span style='color:#fff;font-size:9px;font-weight:700;
                     white-space:nowrap;padding:0 2px;'>{pct_lbl(gp)}</span>
      </div>
      <div style='width:{rp:.1f}%;min-width:{rmin};background:#dc2626;
                  display:flex;align-items:center;justify-content:center;'>
        <span style='color:#fff;font-size:9px;font-weight:700;
                     white-space:nowrap;padding:0 2px;'>{pct_lbl(rp)}</span>
      </div>
    </div>"""

# ═══════════════════════════════════════════════════════════════
#  METRIC CELL BUILDERS
# ═══════════════════════════════════════════════════════════════
def simple_metric_td(pct, pad="6px 8px"):
    col=metric_color(pct)
    bg="#fff5f5" if pct<50 else ("#fff8f0" if pct<70 else "#f0fdf4")
    return f"""
    <td style='padding:{pad};text-align:center;background:{bg};
               border-bottom:1px solid #f1f5f9;border-left:1px solid #f0f4f8;'>
      <span style='font-size:12px;font-weight:700;color:{col};'>{pct}%</span>
    </td>"""

def otr_region_td(pct, above50, below_eq50, total_s, pad="6px 7px", bar_width=80):
    col=metric_color(pct)
    bg="#fff5f5" if pct<50 else ("#fff8f0" if pct<70 else "#f0fdf4")
    bar_html=""
    if total_s>0:
        ab_pct=round(above50/total_s*100,1)
        be_pct=round(below_eq50/total_s*100,1)
        bar_html=f"""
        <div style='display:flex;gap:3px;justify-content:center;
                    margin-top:2px;flex-wrap:nowrap;'>
          <span style='font-size:9px;color:#16a34a;font-weight:600;
                       background:#f0fdf4;padding:1px 4px;border-radius:3px;
                       white-space:nowrap;'>&gt;50%:{above50}</span>
          <span style='font-size:9px;color:#dc2626;font-weight:600;
                       background:#fef2f2;padding:1px 4px;border-radius:3px;
                       white-space:nowrap;'>≤50%:{below_eq50}</span>
        </div>
        <div style='margin-top:2px;text-align:center;'>
          {split_bar(ab_pct,be_pct,above50,below_eq50,width=bar_width)}
        </div>"""
    return f"""
    <td style='padding:{pad};text-align:center;background:{bg};
               border-bottom:1px solid #f1f5f9;border-left:1px solid #f0f4f8;'>
      <span style='font-size:12px;font-weight:700;color:{col};'>{pct}%</span>
      {bar_html}
    </td>"""

def metric_th(label, sublabel, min_width="72px"):
    return f"""
    <th style='padding:7px 8px;text-align:center;min-width:{min_width};
               color:#475569;font-weight:600;border-bottom:2px solid #e2e8f0;'>
      <div style='font-size:11px;'>{label}</div>
      <div style='font-size:9px;color:#94a3b8;font-weight:400;margin-top:1px;'>
        {sublabel}
      </div>
    </th>"""

# ═══════════════════════════════════════════════════════════════
#  STAGE CELL BUILDERS
# ═══════════════════════════════════════════════════════════════
def get_stage_pct_orders(df, stage, store=None):
    col=f'{stage}_secs'; limit=SLA_CONFIG[stage]['hard_limit']
    ds=df if store is None else df[df['Store']==store]
    if col not in ds.columns: return 0,0,0,0.0,0.0
    total=len(ds)
    if total==0: return 0,0,0,0.0,0.0
    bc=int((ds[col]>limit).sum()); wc=total-bc
    return total,wc,bc,round(wc/total*100,1),round(bc/total*100,1)

def get_stage_pct_stores(stores_agg_df, stage):
    col=f'{stage}_secs'; limit=SLA_CONFIG[stage]['hard_limit']
    if col not in stores_agg_df.columns or stores_agg_df.empty:
        return 0,0,0,0.0,0.0
    valid=stores_agg_df[stores_agg_df[col]>0]; total=len(valid)
    if total==0: return 0,0,0,0.0,0.0
    bc=int((valid[col]>limit).sum()); wc=total-bc
    return total,wc,bc,round(wc/total*100,1),round(bc/total*100,1)

def build_stage_cell_region(stage, avg, stores_agg_df, bar_width=105):
    limit=SLA_CONFIG[stage]['hard_limit']
    is_breach=avg>limit and avg>0
    sev=severity_label(avg,limit) if is_breach else "OK"
    sc=sev_color(sev); cell_bg="#fff5f5" if is_breach else "#fafafa"
    wt="700" if is_breach else "500"
    total_,wn,bn,wp,bp=get_stage_pct_stores(stores_agg_df,stage)
    if total_>0:
        counts_html=f"""
        <div style='display:flex;gap:3px;justify-content:center;
                    margin-top:2px;flex-wrap:nowrap;'>
          <span style='font-size:9px;color:#16a34a;font-weight:600;
                       background:#f0fdf4;padding:1px 4px;border-radius:3px;
                       white-space:nowrap;'>OK:{wn}</span>
          <span style='font-size:9px;color:#dc2626;font-weight:600;
                       background:#fef2f2;padding:1px 4px;border-radius:3px;
                       white-space:nowrap;'>B:{bn}</span>
        </div>
        <div style='margin-top:2px;text-align:center;'>
          {split_bar(wp,bp,wn,bn,width=bar_width)}
        </div>"""
    else:
        counts_html="<div style='font-size:9px;color:#cbd5e1;margin-top:2px;'>—</div>"
    return f"""
    <td style='padding:6px 6px;text-align:center;background:{cell_bg};
               border-bottom:1px solid #f1f5f9;border-left:1px solid #f0f4f8;'>
      <div style='font-size:11px;font-weight:{wt};color:{sc};line-height:1.2;'>
        {to_readable(avg)}
      </div>
      {counts_html}
    </td>"""

def build_stage_cell_store(stage, avg, raw_df_subset, bar_width=95):
    limit=SLA_CONFIG[stage]['hard_limit']
    is_breach=avg>limit and avg>0
    sev=severity_label(avg,limit) if is_breach else "OK"
    sc=sev_color(sev); cell_bg="#fff5f5" if is_breach else "#fafafa"
    wt="700" if is_breach else "500"
    total_,wn,bn,wp,bp=get_stage_pct_orders(raw_df_subset,stage)
    if total_>0:
        counts_html=f"""
        <div style='display:flex;gap:3px;justify-content:center;
                    margin-top:2px;flex-wrap:nowrap;'>
          <span style='font-size:9px;color:#16a34a;font-weight:600;
                       background:#f0fdf4;padding:1px 4px;border-radius:3px;
                       white-space:nowrap;'>OK:{wn}</span>
          <span style='font-size:9px;color:#dc2626;font-weight:600;
                       background:#fef2f2;padding:1px 4px;border-radius:3px;
                       white-space:nowrap;'>B:{bn}</span>
        </div>
        <div style='margin-top:2px;text-align:center;'>
          {split_bar(wp,bp,wn,bn,width=bar_width)}
        </div>"""
    else:
        counts_html="<div style='font-size:9px;color:#cbd5e1;margin-top:2px;'>—</div>"
    return f"""
    <td style='padding:6px 6px;text-align:center;background:{cell_bg};
               border-bottom:1px solid #f1f5f9;border-left:1px solid #f0f4f8;'>
      <div style='font-size:11px;font-weight:{wt};color:{sc};line-height:1.2;'>
        {to_readable(avg)}
      </div>
      {counts_html}
    </td>"""

# ═══════════════════════════════════════════════════════════════
#  LOAD & FILTER
# ═══════════════════════════════════════════════════════════════
def load_dump(fb):
    df=pd.read_excel(io.BytesIO(fb),sheet_name='Sheet1',header=0)
    df.columns=[str(c).strip() for c in df.columns]
    df=df.rename(columns={'sa_name':'Store'})
    if 'OTR' in df.columns:
        df['OTR']=df['OTR'].apply(normalize_otr_val)
    else:
        df['OTR']='0'
    if 'MOD' in df.columns:
        df['MOD']=df['MOD'].apply(normalize_mod_val)
    else:
        df['MOD']='0'
    df['_otr'] =df['OTR Status'].astype(str).str.strip().str.lower()
    df['_slak']=df['SLA Key'].astype(str).str.strip()

    all_del=df[df['_otr']=='delivered'].copy()
    all_del=all_del[all_del['Store'].notna()]
    all_del=all_del[all_del['Store'].astype(str).str.strip().str.startswith('DS-')]
    all_del['Region']=all_del['Store'].apply(extract_region)
    all_del['Tier']  =all_del['Region'].apply(get_tier)
    all_delivered_df=all_del[['Store','OTR','MOD','Region','Tier']].reset_index(drop=True)

    before=len(df)
    df_main=df[(df['_otr']=='delivered')&(df['_slak']=='Yes')].copy()
    after=len(df_main)
    df_main.drop(columns=['_otr','_slak'],inplace=True)
    df_main=df_main.rename(columns=DUMP_COL_MAP)
    df_main=df_main[df_main['Store'].notna()]
    df_main=df_main[df_main['Store'].astype(str).str.strip().str.startswith('DS-')]
    df_main=df_main.reset_index(drop=True)

    for col in STAGE_COLS:
        df_main[f'{col}_secs']=df_main[col].apply(time_to_seconds) if col in df_main.columns else 0
    df_main['InhouseSLA_secs']=(df_main['OP to INP_secs']+
                                 df_main['INP to PK_secs']+
                                 df_main['PK to BIN_secs'])
    df_main['Region']=df_main['Store'].apply(extract_region)
    df_main['Tier']  =df_main['Region'].apply(get_tier)
    return df_main, all_delivered_df, before, after

# ═══════════════════════════════════════════════════════════════
#  AGGREGATE
# ═══════════════════════════════════════════════════════════════
def aggregate_by_store(df):
    records=[]
    for store,grp in df.groupby('Store'):
        row={'Store':store,'Region':grp['Region'].iloc[0],
             'Tier':grp['Tier'].iloc[0],'Total_Orders':len(grp)}
        for col in INDIVIDUAL_SECS:
            nz=grp[grp[col]>0][col]
            row[col]=round(nz.mean(),4) if len(nz)>0 else 0.0
        row['InhouseSLA_secs']=(row.get('OP to INP_secs',0)+
                                 row.get('INP to PK_secs',0)+
                                 row.get('PK to BIN_secs',0))
        records.append(row)
    return pd.DataFrame(records)

def aggregate_by_region(df):
    records=[]
    for region,grp in df.groupby('Region'):
        if region in ('Unknown','#N/A','','nan'): continue
        row={'Region':region,'Tier':grp['Tier'].iloc[0],'Total_Orders':len(grp)}
        for col in INDIVIDUAL_SECS:
            nz=grp[grp[col]>0][col]
            row[col]=round(nz.mean(),4) if len(nz)>0 else 0.0
        row['InhouseSLA_secs']=(row.get('OP to INP_secs',0)+
                                 row.get('INP to PK_secs',0)+
                                 row.get('PK to BIN_secs',0))
        records.append(row)
    return pd.DataFrame(records)

def flag_breaches(df):
    for stage in list(SLA_CONFIG.keys()):
        col=f'{stage}_secs'; limit=SLA_CONFIG[stage]['hard_limit']
        if col in df.columns:
            df[f'{stage}_breach']=(df[col]>limit)&(df[col]>0)
        else:
            df[f'{stage}_breach']=False
    return df

# ═══════════════════════════════════════════════════════════════
#  TIER KPI CARDS
# ═══════════════════════════════════════════════════════════════
def tier_kpi_html(region_df, store_df, raw_df, all_delivered_df):
    TIER_META={
        'T1':{'label':'Tier 1 — Metro',       'color':'#1e293b'},
        'T2':{'label':'Tier 2 — Major Cities', 'color':'#1e3a5f'},
        'T3':{'label':'Tier 3 — Rural',        'color':'#1e4a3f'},
    }
    html=""
    for tier in ['T1','T2','T3']:
        t_regions=region_df[region_df['Tier']==tier]
        t_stores =store_df[store_df['Tier']==tier]
        t_raw    =raw_df[raw_df['Tier']==tier]
        if t_regions.empty: continue
        s_high=0;s_medium=0;s_ok=0;s_ih=0;s_br=0
        for _,sr in t_stores.iterrows():
            p=get_priority(sr)
            if p=='HIGH':     s_high  +=1
            elif p=='MEDIUM': s_medium+=1
            else:             s_ok    +=1
            if sr.get('InhouseSLA_breach',False): s_ih+=1
            if sr.get('BIN to RTS_breach',False): s_br+=1
        tier_otr=compute_otr_pct(all_delivered_df,tier=tier)
        tier_mod=compute_mod_pct(all_delivered_df,tier=tier)
        otr_col=metric_color(tier_otr); mod_col=metric_color(tier_mod)
        meta=TIER_META[tier]

        def std_card(val,lbl,col,bl=True):
            border="border-left:1px solid #f1f5f9;" if bl else ""
            return f"""
            <div style='flex:1;padding:10px 6px;text-align:center;{border}'>
              <div style='font-size:18px;font-weight:700;color:{col};line-height:1.2;'>{val}</div>
              <div style='font-size:9px;color:#94a3b8;margin-top:2px;white-space:nowrap;'>{lbl}</div>
            </div>"""

        otr_card=f"""<div style='flex:1;padding:10px 6px;text-align:center;border-left:1px solid #f1f5f9;'>
          <div style='font-size:18px;font-weight:700;color:{otr_col};line-height:1.2;'>{tier_otr}%</div>
          <div style='font-size:9px;color:#94a3b8;margin-top:2px;'>OTR %</div>
        </div>"""
        mod_card=f"""<div style='flex:1;padding:10px 6px;text-align:center;border-left:1px solid #f1f5f9;'>
          <div style='font-size:18px;font-weight:700;color:{mod_col};line-height:1.2;'>{tier_mod}%</div>
          <div style='font-size:9px;color:#94a3b8;margin-top:2px;'>MOD %</div>
        </div>"""

        html+=f"""
        <div style='margin-bottom:14px;'>
          <div style='background:{meta["color"]};color:#fff;border-radius:8px 8px 0 0;
                      padding:6px 14px;font-size:11px;font-weight:700;letter-spacing:0.3px;'>
            {meta["label"]} &nbsp;·&nbsp;
            <span style='opacity:0.75;font-weight:400;'>
              {len(t_regions)} regions &nbsp;·&nbsp;
              {len(t_stores)} stores &nbsp;·&nbsp; {len(t_raw):,} orders
            </span>
          </div>
          <div style='display:flex;gap:0;background:#fff;border:1px solid #e2e8f0;
                      border-top:none;border-radius:0 0 8px 8px;overflow:hidden;'>
            {std_card(len(t_regions),"Regions","#1e293b",False)}
            {std_card(len(t_stores),"Stores","#334155")}
            {std_card(f"{len(t_raw):,}","Orders","#475569")}
            {otr_card}
            {std_card(s_high,"HIGH Stores","#dc2626")}
            {std_card(s_medium,"MEDIUM Stores","#ea580c")}
            {std_card(s_ok,"Stores OK","#16a34a")}
            {std_card(s_ih,"IH-SLA Breach","#ea580c")}
            {std_card(s_br,"BIN→RTS Breach","#d97706")}
            {mod_card}
          </div>
        </div>"""
    return html

# ═══════════════════════════════════════════════════════════════
#  SHARED COLUMN HEADER ROW BUILDERS
# ═══════════════════════════════════════════════════════════════
def summary_col_headers():
    STAGE_SHORT={
        'OP to INP':'OP → INP','INP to PK':'INP → PK',
        'PK to BIN':'PK → BIN','InhouseSLA':'InhouseSLA','BIN to RTS':'BIN → RTS',
    }
    stage_th="".join(f"""
        <th style='padding:7px 10px;text-align:center;min-width:100px;color:#475569;
                   font-weight:600;border-bottom:2px solid #e2e8f0;'>
          <div style='font-size:11px;'>{STAGE_SHORT[s]}</div>
          <div style='font-size:9px;color:#94a3b8;font-weight:400;margin-top:1px;'>
            Limit: {to_readable(SLA_CONFIG[s]['hard_limit'])}
          </div>
        </th>""" for s in SUMMARY_STAGES)
    return f"""
    <tr style='background:#f1f5f9;'>
      <th style='padding:7px 12px;text-align:left;color:#475569;font-weight:600;
                 border-bottom:2px solid #e2e8f0;min-width:170px;'>
        Region
        <div style='font-size:9px;color:#94a3b8;font-weight:400;margin-top:1px;'>
          HIGH → MEDIUM → OK &nbsp;·&nbsp; InhouseSLA ↓
        </div>
      </th>
      {metric_th("OTR %","OTR=10","65px")}
      {stage_th}
      {metric_th("MOD %","MOD=1","65px")}
      <th style='padding:7px 10px;text-align:center;border-bottom:2px solid #e2e8f0;
                 min-width:75px;color:#475569;font-weight:600;'>Status</th>
    </tr>"""

def breakdown_col_headers():
    STAGE_SHORT={
        'OP to INP':'OP → INP','INP to PK':'INP → PK',
        'PK to BIN':'PK → BIN','InhouseSLA':'InhouseSLA','BIN to RTS':'BIN → RTS',
    }
    stage_th="".join(f"""
        <th style='padding:7px 7px;text-align:center;min-width:120px;'>
          <div style='font-weight:700;font-size:11px;'>{STAGE_SHORT[s]}</div>
          <div style='font-size:9px;color:#94a3b8;font-weight:400;margin-top:1px;'>
            Limit: {to_readable(SLA_CONFIG[s]['hard_limit'])}
          </div>
        </th>""" for s in BREAKDOWN_STAGES)
    return f"""
    <tr style='background:#f1f5f9;'>
      <th style='padding:7px 12px;text-align:left;color:#475569;font-weight:700;
                 border-bottom:2px solid #e2e8f0;min-width:180px;'>
        Region
        <div style='font-size:9px;color:#94a3b8;font-weight:400;margin-top:1px;'>
          HIGH → MEDIUM · InhouseSLA ↓ · stores split
        </div>
      </th>
      {metric_th("OTR %","OTR=10 · &gt;50% / ≤50%","88px")}
      {stage_th}
      {metric_th("MOD %","MOD=1","65px")}
      <th style='padding:7px 7px;text-align:center;border-bottom:2px solid #e2e8f0;
                 min-width:75px;'>Status</th>
    </tr>"""

# ═══════════════════════════════════════════════════════════════
#  SUMMARY REPORT  (one table per tier)
# ═══════════════════════════════════════════════════════════════
def summary_report_html(region_df, raw_df, all_delivered_df):
    TIER_LABELS={'T1':'Tier 1 — Metro','T2':'Tier 2 — Major Cities','T3':'Tier 3 — Rural'}
    TIER_BG={'T1':'#1e293b','T2':'#1e3a5f','T3':'#1e4a3f'}

    html="<div style='overflow-x:auto;'>"

    for tier in ['T1','T2','T3']:
        t_regions=region_df[region_df['Tier']==tier].copy()
        if t_regions.empty: continue
        t_regions['_priority']=t_regions.apply(get_priority,axis=1)
        t_regions['_psort']   =t_regions['_priority'].map(priority_sort_key)
        t_regions=t_regions.sort_values(
            ['_psort','InhouseSLA_secs'],ascending=[True,False]
        ).reset_index(drop=True)

        n_high  =int((t_regions['_priority']=='HIGH').sum())
        n_medium=int((t_regions['_priority']=='MEDIUM').sum())
        n_ok    =int((t_regions['_priority']=='OK').sum())
        tier_otr=compute_otr_pct(all_delivered_df,tier=tier)
        tier_mod=compute_mod_pct(all_delivered_df,tier=tier)

        html+=f"""
        <table style='width:100%;border-collapse:collapse;font-size:11px;
                      background:#fff;border-radius:10px;overflow:hidden;
                      border:1px solid #e2e8f0;margin-bottom:16px;'>
          <thead>
            <tr>
              <td colspan='{TOTAL_COLS}'
                  style='background:{TIER_BG[tier]};color:#fff;font-size:11px;
                         font-weight:700;padding:7px 14px;letter-spacing:0.3px;'>
                {TIER_LABELS[tier]} &nbsp;·&nbsp; {len(t_regions)} regions
                &nbsp;&nbsp;
                <span style='color:#fca5a5;font-size:10px;'>■ {n_high} HIGH</span>
                &nbsp;<span style='color:#fdba74;font-size:10px;'>■ {n_medium} MEDIUM</span>
                &nbsp;<span style='color:#86efac;font-size:10px;'>■ {n_ok} OK</span>
                &nbsp;&nbsp;
                <span style='color:#a5f3fc;font-size:10px;'>OTR:{tier_otr}%</span>
                &nbsp;<span style='color:#c4b5fd;font-size:10px;'>MOD:{tier_mod}%</span>
              </td>
            </tr>
            {summary_col_headers()}
          </thead>
          <tbody>"""

        for _,row in t_regions.iterrows():
            region_name=row['Region']
            priority   =row['_priority']
            r_raw      =raw_df[raw_df['Region']==region_name]
            region_otr =compute_otr_pct(all_delivered_df,region=region_name)
            region_mod =compute_mod_pct(all_delivered_df,region=region_name)

            p_styles={
                "HIGH":  ("#dc2626","#fef2f2","#fecaca"),
                "MEDIUM":("#ea580c","#fff7ed","#fed7aa"),
                "OK":    ("#16a34a","#f0fdf4","#bbf7d0"),
            }
            pc,pbg,pborder=p_styles.get(priority,("#64748b","#f8fafc","#e2e8f0"))
            status_badge=f"""
            <span style='background:{pbg};color:{pc};border:1px solid {pborder};
                         border-radius:4px;padding:2px 7px;font-size:10px;font-weight:700;
                         white-space:nowrap;display:inline-block;'>{priority}</span>"""
            row_bg={"HIGH":"#fffcfc","MEDIUM":"#fffdf9","OK":"#ffffff"}.get(priority,"#ffffff")

            stage_cells=""
            for stage in SUMMARY_STAGES:
                col=f'{stage}_secs'; limit=SLA_CONFIG[stage]['hard_limit']
                avg=row.get(col,0); is_breach=avg>limit and avg>0
                sev=severity_label(avg,limit) if is_breach else "OK"
                sc=sev_color(sev); cell_bg="#fff5f5" if is_breach else "#fff"
                wt="700" if is_breach else "400"
                stage_cells+=f"""
                <td style='padding:6px 8px;text-align:center;background:{cell_bg};
                           border-bottom:1px solid #f1f5f9;border-left:1px solid #f0f4f8;'>
                  <span style='font-size:11px;font-weight:{wt};color:{sc};'>
                    {to_readable(avg)}
                  </span>
                </td>"""

            html+=f"""
          <tr style='background:{row_bg};'>
            <td style='padding:6px 12px;font-size:11px;font-weight:600;color:#1e293b;
                       border-bottom:1px solid #f1f5f9;border-left:3px solid {pc};
                       white-space:nowrap;'>
              {region_name}
              <div style='font-size:9px;color:#94a3b8;font-weight:400;margin-top:1px;'>
                {len(r_raw):,} orders
              </div>
            </td>
            {simple_metric_td(region_otr,"6px 8px")}
            {stage_cells}
            {simple_metric_td(region_mod,"6px 8px")}
            <td style='padding:6px 8px;text-align:center;border-bottom:1px solid #f1f5f9;'>
              {status_badge}
            </td>
          </tr>"""

        avg_cells=""
        for stage in SUMMARY_STAGES:
            col=f'{stage}_secs'; limit=SLA_CONFIG[stage]['hard_limit']
            vv=t_regions[t_regions[col]>0][col]
            tavg=round(vv.mean(),4) if len(vv)>0 else 0.0
            is_breach=tavg>limit and tavg>0
            sev=severity_label(tavg,limit) if is_breach else "OK"
            sc=sev_color(sev); cell_bg="#fef3f2" if is_breach else "#f0fdf4"
            avg_cells+=f"""
            <td style='padding:6px 8px;text-align:center;background:{cell_bg};
                       border-bottom:2px solid #e2e8f0;border-left:1px solid #e2e8f0;'>
              <span style='font-size:11px;font-weight:700;color:{sc};'>{to_readable(tavg)}</span>
              <div style='font-size:8px;color:#94a3b8;margin-top:1px;'>avg</div>
            </td>"""

        otr_c=metric_color(tier_otr)
        otr_bg="#fff5f5" if tier_otr<50 else ("#fff8f0" if tier_otr<70 else "#f0fdf4")
        mod_c=metric_color(tier_mod)
        mod_bg="#fff5f5" if tier_mod<50 else ("#fff8f0" if tier_mod<70 else "#f0fdf4")

        html+=f"""
          <tr style='background:#f8fafc;'>
            <td style='padding:6px 12px;font-size:11px;font-weight:700;color:#334155;
                       border-bottom:2px solid #e2e8f0;border-left:3px solid {TIER_BG[tier]};
                       white-space:nowrap;'>
              &#8709; Tier Average
              <div style='font-size:9px;color:#94a3b8;font-weight:400;margin-top:1px;'>
                across {len(t_regions)} regions
              </div>
            </td>
            <td style='padding:6px 8px;text-align:center;background:{otr_bg};
                       border-bottom:2px solid #e2e8f0;border-left:1px solid #e2e8f0;'>
              <span style='font-size:11px;font-weight:700;color:{otr_c};'>{tier_otr}%</span>
              <div style='font-size:8px;color:#94a3b8;margin-top:1px;'>avg</div>
            </td>
            {avg_cells}
            <td style='padding:6px 8px;text-align:center;background:{mod_bg};
                       border-bottom:2px solid #e2e8f0;border-left:1px solid #e2e8f0;'>
              <span style='font-size:11px;font-weight:700;color:{mod_c};'>{tier_mod}%</span>
              <div style='font-size:8px;color:#94a3b8;margin-top:1px;'>avg</div>
            </td>
            <td style='padding:6px 8px;text-align:center;border-bottom:2px solid #e2e8f0;
                       font-size:10px;color:#94a3b8;'>—</td>
          </tr>
        </tbody></table>"""

    html+="</div>"
    return html

# ═══════════════════════════════════════════════════════════════
#  STORE BREAKDOWN TABLE
# ═══════════════════════════════════════════════════════════════
def store_breakdown_html(region_stores_agg, region_raw, all_delivered_df):
    STAGE_SHORT={
        'OP to INP':'OP → INP','INP to PK':'INP → PK',
        'PK to BIN':'PK → BIN','InhouseSLA':'InhouseSLA','BIN to RTS':'BIN → RTS',
    }
    if region_stores_agg.empty:
        return "<p style='color:#94a3b8;font-size:12px;padding:6px;'>No store data.</p>"
    store_rows=[(sr,get_priority(sr)) for _,sr in region_stores_agg.iterrows()
                if get_priority(sr)!='OK']
    if not store_rows:
        return """<div style='padding:8px 12px;font-size:11px;color:#16a34a;
                              background:#f0fdf4;border-radius:6px;border:1px solid #bbf7d0;'>
                    ✓ All stores within SLA
                  </div>"""
    store_rows.sort(key=lambda x:(priority_sort_key(x[1]),-x[0].get('InhouseSLA_secs',0)))

    stage_th="".join(f"""
        <th style='padding:6px 6px;text-align:center;min-width:110px;font-size:10px;'>
          <div style='font-weight:700;'>{STAGE_SHORT[s]}</div>
          <div style='font-size:9px;color:#94a3b8;font-weight:400;'>
            Limit:{to_readable(SLA_CONFIG[s]['hard_limit'])}
          </div>
        </th>""" for s in BREAKDOWN_STAGES)

    html=f"""
    <div style='overflow-x:auto;margin-top:4px;'>
    <table style='width:100%;border-collapse:collapse;font-size:11px;
                  background:#fff;border-radius:8px;overflow:hidden;border:1px solid #e2e8f0;'>
      <thead>
        <tr style='background:#f8fafc;'>
          <th style='padding:6px 10px;text-align:left;color:#475569;font-weight:700;
                     border-bottom:2px solid #e2e8f0;min-width:190px;font-size:10px;'>
            Store (HIGH → MEDIUM · InhouseSLA ↓)
          </th>
          {stage_th}
          {metric_th("OTR %","OTR=10","58px")}
          {metric_th("MOD %","MOD=1","58px")}
          <th style='padding:6px 6px;text-align:center;border-bottom:2px solid #e2e8f0;
                     min-width:65px;font-size:10px;'>Status</th>
        </tr>
      </thead>
      <tbody>"""

    for sr,priority in store_rows:
        store_name =sr['Store']
        s_raw      =region_raw[region_raw['Store']==store_name]
        inhouse_val=sr.get('InhouseSLA_secs',0)
        store_otr  =compute_otr_pct(all_delivered_df,store=store_name)
        store_mod  =compute_mod_pct(all_delivered_df,store=store_name)
        p_styles={
            "HIGH":  ("#dc2626","#fef2f2","#fecaca"),
            "MEDIUM":("#ea580c","#fff7ed","#fed7aa"),
        }
        pc,pbg,pborder=p_styles.get(priority,("#64748b","#f8fafc","#e2e8f0"))
        status_badge=f"""<span style='background:{pbg};color:{pc};border:1px solid {pborder};
                     border-radius:4px;padding:2px 6px;font-size:9px;
                     font-weight:700;white-space:nowrap;'>{priority}</span>"""
        row_bg="#fffcfc" if priority=="HIGH" else "#fffdf9"
        stage_cells="".join(
            build_stage_cell_store(stage,sr.get(f'{stage}_secs',0),s_raw,bar_width=95)
            for stage in BREAKDOWN_STAGES)
        html+=f"""
      <tr style='background:{row_bg};'>
        <td style='padding:6px 10px;font-size:10px;font-weight:600;color:#1e293b;
                   border-bottom:1px solid #f1f5f9;white-space:nowrap;
                   border-left:3px solid {pc};'>
          {store_name}
          <div style='font-size:9px;color:#94a3b8;font-weight:400;margin-top:1px;'>
            {len(s_raw):,} orders &nbsp;·&nbsp; IH-SLA: {to_readable(inhouse_val)}
          </div>
        </td>
        {stage_cells}
        {simple_metric_td(store_otr,"6px 6px")}
        {simple_metric_td(store_mod,"6px 6px")}
        <td style='padding:6px 6px;text-align:center;border-bottom:1px solid #f1f5f9;'>
          {status_badge}
        </td>
      </tr>"""

    html+="</tbody></table></div>"
    return html

# ═══════════════════════════════════════════════════════════════
#  REGIONWISE BREAKDOWN  (one table per tier)
# ═══════════════════════════════════════════════════════════════
def regionwise_breakdown_html(region_df, store_df, raw_df, all_delivered_df):
    TIER_LABELS={'T1':'Tier 1 — Metro','T2':'Tier 2 — Major Cities','T3':'Tier 3 — Rural'}
    TIER_BG={'T1':'#1e293b','T2':'#1e3a5f','T3':'#1e4a3f'}

    uid_counter=[0]
    total_shown=0
    html="<div style='overflow-x:auto;'>"

    for tier in ['T1','T2','T3']:
        t_regions=region_df[region_df['Tier']==tier].copy()
        if t_regions.empty: continue
        t_regions['_priority']=t_regions.apply(get_priority,axis=1)
        t_breach=t_regions[t_regions['_priority']!='OK'].copy()
        if t_breach.empty: continue
        t_breach['_psort']=t_breach['_priority'].map(priority_sort_key)
        t_breach=t_breach.sort_values(
            ['_psort','InhouseSLA_secs'],ascending=[True,False]
        ).reset_index(drop=True)
        n_high  =int((t_breach['_priority']=='HIGH').sum())
        n_medium=int((t_breach['_priority']=='MEDIUM').sum())

        html+=f"""
        <table style='width:100%;border-collapse:collapse;font-size:11px;
                      background:#fff;border-radius:10px;overflow:hidden;
                      border:1px solid #e2e8f0;margin-bottom:16px;'>
          <thead>
            <tr>
              <td colspan='{TOTAL_COLS}'
                  style='background:{TIER_BG[tier]};color:#fff;font-size:11px;
                         font-weight:700;padding:7px 14px;letter-spacing:0.3px;'>
                {TIER_LABELS[tier]} &nbsp;·&nbsp; {len(t_breach)} breaching
                &nbsp;&nbsp;
                <span style='color:#fca5a5;font-size:10px;'>■ {n_high} HIGH</span>
                &nbsp;<span style='color:#fdba74;font-size:10px;'>■ {n_medium} MEDIUM</span>
                &nbsp;&nbsp;
                <span style='color:#86efac;font-size:10px;font-weight:400;'>
                  (sorted by InhouseSLA ↓)
                </span>
              </td>
            </tr>
            {breakdown_col_headers()}
          </thead>
          <tbody>"""

        for _,region_row_data in t_breach.iterrows():
            uid_counter[0]+=1
            uid        =uid_counter[0]
            region_name=region_row_data['Region']
            priority   =region_row_data['_priority']
            inhouse_val=region_row_data.get('InhouseSLA_secs',0)
            r_raw      =raw_df[raw_df['Region']==region_name]
            r_stores   =store_df[store_df['Region']==region_name]
            region_otr =compute_otr_pct(all_delivered_df,region=region_name)
            region_mod =compute_mod_pct(all_delivered_df,region=region_name)
            ts_otr,ab_otr,be_otr=compute_otr_store_brackets(
                all_delivered_df,region=region_name)
            total_shown+=1

            p_styles={
                "HIGH":  ("#dc2626","#fef2f2","#fecaca"),
                "MEDIUM":("#ea580c","#fff7ed","#fed7aa"),
            }
            pc,pbg,pborder=p_styles.get(priority,("#64748b","#f8fafc","#e2e8f0"))
            status_badge=f"""
            <span style='background:{pbg};color:{pc};border:1px solid {pborder};
                         border-radius:4px;padding:2px 7px;font-size:10px;font-weight:700;
                         white-space:nowrap;display:inline-block;'>{priority}</span>"""
            row_bg="#fffcfc" if priority=="HIGH" else "#fffdf9"

            stage_cells="".join(
                build_stage_cell_region(stage,region_row_data.get(f'{stage}_secs',0),
                                        r_stores,bar_width=105)
                for stage in BREAKDOWN_STAGES)

            r_stores_breach=sum(1 for _,sr in r_stores.iterrows() if get_priority(sr)!='OK')
            store_html=store_breakdown_html(r_stores,r_raw,all_delivered_df)
            otr_cell=otr_region_td(region_otr,ab_otr,be_otr,ts_otr,
                                    pad="6px 7px",bar_width=80)
            mod_cell=simple_metric_td(region_mod,"6px 7px")

            html+=f"""
          <tr style='background:{row_bg};' id='rrow-{uid}'>
            <td style='padding:0;border-bottom:1px solid #f1f5f9;border-left:3px solid {pc};'>
              <details id='rdet-{uid}' style='margin:0;'>
                <summary style='list-style:none;cursor:pointer;padding:7px 12px;
                                display:flex;align-items:flex-start;gap:8px;
                                user-select:none;flex-wrap:wrap;'>
                  <span id='rarr-{uid}' style='font-size:10px;color:#94a3b8;
                                               margin-top:1px;min-width:12px;'>▶</span>
                  <div>
                    <div style='font-size:12px;font-weight:700;color:#1e293b;line-height:1.2;'>
                      {region_name}
                    </div>
                    <div style='font-size:9px;color:#94a3b8;margin-top:2px;'>
                      {len(r_raw):,} orders &nbsp;·&nbsp;
                      {len(r_stores)} stores ({r_stores_breach} breaching)
                      &nbsp;·&nbsp; IH-SLA: {to_readable(inhouse_val)}
                    </div>
                  </div>
                </summary>
                <div style='padding:8px 12px 12px;background:#f8fafc;
                            border-top:1px dashed #e2e8f0;'>
                  <div style='font-size:10px;font-weight:700;color:#475569;margin-bottom:6px;'>
                    &#128203; Store Breakdown &nbsp;·&nbsp; Breaching stores only
                    &nbsp;·&nbsp; HIGH → MEDIUM · InhouseSLA ↓
                  </div>
                  {store_html}
                </div>
              </details>
            </td>
            {otr_cell}
            {stage_cells}
            {mod_cell}
            <td style='padding:6px 7px;text-align:center;border-bottom:1px solid #f1f5f9;'>
              {status_badge}
            </td>
          </tr>
          <script>
          (function(){{
            var d=document.getElementById('rdet-{uid}');
            var a=document.getElementById('rarr-{uid}');
            if(d&&a){{d.addEventListener('toggle',function(){{
              a.textContent=d.open?'▼':'▶';
            }});}}
          }})();
          </script>"""

        html+="</tbody></table>"

    if total_shown==0:
        html+="""<div style='padding:20px;text-align:center;color:#16a34a;
                              font-size:13px;font-weight:600;'>
          ✓ All regions are within SLA — No breaches found
        </div>"""

    html+="</div>"
    return html

# ═══════════════════════════════════════════════════════════════
#  QUICK SUMMARY SNAPSHOT
# ═══════════════════════════════════════════════════════════════
def generate_summary_snapshot_html(region_df, store_df, raw_df, all_delivered_df):
    now = datetime.now().strftime("%d %b %Y, %I:%M %p")

    total_orders  = len(raw_df)
    total_regions = len(region_df[~region_df['Region'].isin(['Unknown',''])])
    total_stores  = len(store_df)
    overall_otr   = compute_otr_pct(all_delivered_df)
    overall_mod   = compute_mod_pct(all_delivered_df)

    region_df2 = region_df.copy()
    region_df2['_priority'] = region_df2.apply(get_priority, axis=1)
    store_df2  = store_df.copy()
    store_df2['_priority']  = store_df2.apply(get_priority, axis=1)

    high_regions   = region_df2[region_df2['_priority']=='HIGH']
    medium_regions = region_df2[region_df2['_priority']=='MEDIUM']
    ok_regions     = region_df2[region_df2['_priority']=='OK']
    high_stores    = store_df2[store_df2['_priority']=='HIGH']
    medium_stores  = store_df2[store_df2['_priority']=='MEDIUM']

    stage_breach_info = {}
    for stage in SUMMARY_STAGES:
        col   = f'{stage}_secs'
        limit = SLA_CONFIG[stage]['hard_limit']
        if col not in region_df2.columns:
            stage_breach_info[stage] = {'count':0,'pct':0,'top':[],'avg_excess':0}
            continue
        breaching  = region_df2[region_df2[col] > limit]
        bc         = len(breaching)
        pct        = round(bc / max(total_regions,1) * 100)
        top3       = breaching.sort_values(col,ascending=False).head(3)['Region'].tolist()
        excess     = breaching[col] - limit
        avg_excess = round(excess.mean()) if len(excess)>0 else 0
        stage_breach_info[stage] = {'count':bc,'pct':pct,'top':top3,'avg_excess':avg_excess}

    sorted_stages = sorted(stage_breach_info.items(), key=lambda x: -x[1]['count'])
    top_sg        = sorted_stages[0][0] if sorted_stages and sorted_stages[0][1]['count']>0 else None
    top_sg_cnt    = sorted_stages[0][1]['count'] if top_sg else 0

    TIER_BG_MAP   = {'T1':'#1e293b','T2':'#1e3a5f','T3':'#1e4a3f'}
    TIER_FULL_MAP = {'T1':'Tier 1 — Metro','T2':'Tier 2 — Major Cities','T3':'Tier 3 — Rural'}
    tier_data_map = {}

    for tier in ['T1','T2','T3']:
        t_r = region_df2[region_df2['Tier']==tier]
        if t_r.empty:
            tier_data_map[tier] = None
            continue
        t_high   = t_r[t_r['_priority']=='HIGH']
        t_medium = t_r[t_r['_priority']=='MEDIUM']
        t_ok     = t_r[t_r['_priority']=='OK']
        t_s      = store_df2[store_df2['Tier']==tier]
        t_sh     = t_s[t_s['_priority']=='HIGH']
        t_sm     = t_s[t_s['_priority']=='MEDIUM']

        tier_stage_breaches = {}
        for stage in SUMMARY_STAGES:
            col   = f'{stage}_secs'
            limit = SLA_CONFIG[stage]['hard_limit']
            if col in t_r.columns:
                tier_stage_breaches[stage] = int((t_r[col]>limit).sum())

        top_stage       = max(tier_stage_breaches,key=tier_stage_breaches.get) if tier_stage_breaches else None
        top_stage_count = tier_stage_breaches.get(top_stage,0) if top_stage else 0

        t_s_breach  = t_s[t_s['_priority']!='OK'].sort_values('InhouseSLA_secs',ascending=False)
        worst_store = t_s_breach.iloc[0]['Store'] if not t_s_breach.empty else None

        tier_otr_v = compute_otr_pct(all_delivered_df,tier=tier)
        tier_mod_v = compute_mod_pct(all_delivered_df,tier=tier)

        tier_data_map[tier] = {
            'total_regions':       len(t_r),
            'high':                len(t_high),
            'medium':              len(t_medium),
            'ok':                  len(t_ok),
            'total_stores':        len(t_s),
            'high_stores':         len(t_sh),
            'medium_stores':       len(t_sm),
            'top_stage':           top_stage,
            'top_stage_count':     top_stage_count,
            'worst_store':         worst_store,
            'otr':                 tier_otr_v,
            'mod':                 tier_mod_v,
            'raw_orders':          len(raw_df[raw_df['Tier']==tier]),
            'high_region_names':   t_high['Region'].tolist(),
            'medium_region_names': t_medium['Region'].tolist(),
        }

    ACTION_MAP = {
        'OP to INP': {
            'cause':  'Order acceptance & picker assignment delay',
            'impact': 'Picker is taking too long to accept or start the order after it is placed in the OMS.',
            'actions': [
                'Verify picker login status and device connectivity at all affected stores',
                'Audit push-notification delivery lag on store devices and OMS integration',
                'Increase picker headcount during identified peak demand windows',
                'Configure auto-escalation if an order is not accepted within 10 seconds',
                'Cross-check store roster against hourly order volume patterns for HIGH stores',
            ]
        },
        'INP to PK': {
            'cause':  'Picking / item-collection delay inside the store',
            'impact': 'Picker is spending excessive time locating and collecting items across store aisles.',
            'actions': [
                'Audit item slotting and planogram adherence — high-frequency items must be in accessible zones',
                'Flag out-of-stock events that cause picker walkback or substitution loops',
                'Enable zone-based or parallel picking assignment for multi-item orders',
                'Review per-picker productivity metrics and set clear per-item time targets',
                'Verify batch-picking of multiple orders is not delaying individual order completion',
            ]
        },
        'PK to BIN': {
            'cause':  'Packing / binning delay at the packing station',
            'impact': 'Items are picked but not transferred to the designated bin or packing station within SLA.',
            'actions': [
                'Audit packing station congestion and real-time queue depth at HIGH-priority stores',
                'Pre-assign bin slots before pick completion to eliminate handoff wait time',
                'Ensure adequate tote and bag inventory is stocked at all packing stations',
                'Create separate packing lanes for fragile or overweight items',
                'Enforce bin-scan SOP compliance to prevent re-work and double-handling',
            ]
        },
        'InhouseSLA': {
            'cause':  'Combined inhouse SLA breach (OP→INP + INP→PK + PK→BIN exceeding 3m 30s)',
            'impact': 'Total store-side operations are collectively exceeding the 210-second combined threshold.',
            'actions': [
                'Identify the primary sub-stage driver using the stage breakdown tables in this report',
                'Conduct a full store-ops audit for all HIGH-priority stores: staffing, layout, SOP adherence',
                'Implement real-time inhouse SLA alerts on store devices when the 2-minute mark is reached',
                'Initiate an immediate store manager review call for every HIGH-priority store location',
                'Confirm that OMS inhouse SLA config thresholds are correctly set for all affected stores',
            ]
        },
        'BIN to RTS': {
            'cause':  'Delivery partner pickup or store-to-hub transfer delay',
            'impact': 'Order is fully packed and binned but rider or vehicle pickup is significantly delayed.',
            'actions': [
                'Escalate to delivery partner ops team for a tighter pickup SLA at all breaching stores',
                'Audit rider availability, geo-fencing accuracy, and live ETA visibility near each store',
                'Re-align hub-transfer schedules with actual peak order volume time windows',
                'For self-delivery: review routing algorithm efficiency and vehicle availability at peak hours',
                'Add store-level RTS-pending alerts to the live ops dashboard for real-time monitoring',
            ]
        },
    }

    breaching_r = len(high_regions) + len(medium_regions)
    breach_pct  = round(breaching_r / max(total_regions,1) * 100)

    if   breach_pct == 0:  health_label = "✅ Healthy";         health_color = "#16a34a"
    elif breach_pct <= 25: health_label = "⚠️ Minor Issues";   health_color = "#d97706"
    elif breach_pct <= 50: health_label = "🔶 Moderate Stress"; health_color = "#ea580c"
    else:                  health_label = "🔴 Critical";         health_color = "#dc2626"

    otr_c = metric_color(overall_otr)
    mod_c = metric_color(overall_mod)

    def stat_pill(label, value, color, bg):
        return f"""
        <div style='background:{bg};border:1px solid {color}33;border-radius:8px;
                    padding:9px 14px;text-align:center;min-width:82px;flex-shrink:0;'>
          <div style='font-size:17px;font-weight:700;color:{color};line-height:1.2;'>{value}</div>
          <div style='font-size:9px;color:#64748b;margin-top:2px;white-space:nowrap;'>{label}</div>
        </div>"""

    if breaching_r == 0:
        network_para = (
            f"As of {now}, all <b>{total_regions} regions</b> across <b>{total_stores} stores</b> "
            f"are operating within SLA thresholds. The network processed <b>{total_orders:,} "
            f"delivered orders</b> with an overall OTR of "
            f"<b style='color:{otr_c};'>{overall_otr}%</b> and MOD of "
            f"<b style='color:{mod_c};'>{overall_mod}%</b>. No immediate action required."
        )
    else:
        stage_mention = (
            f", with <b>{top_sg}</b> emerging as the most commonly breached stage "
            f"({top_sg_cnt} of {total_regions} regions affected)"
        ) if top_sg else ""
        network_para = (
            f"As of {now}, the SLA health check covers <b>{total_regions} regions</b> and "
            f"<b>{total_stores} stores</b> processing <b>{total_orders:,} delivered orders</b> "
            f"(SLA Key=Yes). The overall OTR stands at "
            f"<b style='color:{otr_c};'>{overall_otr}%</b> and MOD at "
            f"<b style='color:{mod_c};'>{overall_mod}%</b>.<br><br>"
            f"Out of {total_regions} regions, "
            f"<b style='color:#dc2626;'>{len(high_regions)} are HIGH priority</b> "
            f"(both InhouseSLA and BIN→RTS breaching simultaneously), "
            f"<b style='color:#ea580c;'>{len(medium_regions)} are MEDIUM priority</b> "
            f"(single-stage breach), and "
            f"<b style='color:#16a34a;'>{len(ok_regions)} are within SLA</b>{stage_mention}. "
            f"At store level, <b style='color:#dc2626;'>{len(high_stores)} stores are HIGH</b> "
            f"and <b style='color:#ea580c;'>{len(medium_stores)} are MEDIUM</b> — "
            f"requiring immediate operations attention."
        )

    html = f"""
<div style='font-family:Segoe UI,system-ui,Arial,sans-serif;'>

<div style='background:linear-gradient(135deg,#0f172a 0%,#1e293b 100%);
            border-radius:12px;padding:20px 26px;margin-bottom:14px;
            border-left:5px solid #6366f1;'>
  <div style='display:flex;align-items:center;gap:14px;flex-wrap:wrap;'>
    <div>
      <div style='font-size:15px;font-weight:700;color:#fff;letter-spacing:0.3px;'>
        &#128203; Quick Summary Snapshot
      </div>
      <div style='font-size:10px;color:#94a3b8;margin-top:3px;'>
        Auto-generated narrative &nbsp;·&nbsp; {now} &nbsp;·&nbsp;
        {total_orders:,} orders analysed
      </div>
    </div>
    <div style='margin-left:auto;background:{health_color};color:#fff;
                padding:6px 18px;border-radius:20px;font-size:11px;font-weight:700;
                white-space:nowrap;letter-spacing:0.3px;'>
      {health_label}
    </div>
  </div>
</div>

<div style='background:#fff;border:1px solid #e2e8f0;border-radius:10px;
            padding:18px 22px;margin-bottom:14px;border-left:4px solid #6366f1;'>
  <div style='font-size:12px;font-weight:700;color:#1e293b;margin-bottom:10px;
              letter-spacing:0.2px;'>
    &#127758; Network Health Overview
  </div>
  <div style='display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px;'>
    {stat_pill("Total Orders",  f"{total_orders:,}", "#6366f1","#f5f3ff")}
    {stat_pill("Regions",        str(total_regions),  "#0891b2","#ecfeff")}
    {stat_pill("Stores",         str(total_stores),   "#0891b2","#ecfeff")}
    {stat_pill("HIGH Regions",   str(len(high_regions)),  "#dc2626","#fef2f2")}
    {stat_pill("MEDIUM Regions", str(len(medium_regions)),"#ea580c","#fff7ed")}
    {stat_pill("OK Regions",     str(len(ok_regions)),    "#16a34a","#f0fdf4")}
    {stat_pill("Overall OTR%",   f"{overall_otr}%",   otr_c,"#f8fafc")}
    {stat_pill("Overall MOD%",   f"{overall_mod}%",   mod_c,"#f8fafc")}
  </div>
  <p style='font-size:12px;color:#334155;line-height:1.85;margin:0;'>
    {network_para}
  </p>
</div>"""

    html += """
<div style='background:#fff;border:1px solid #e2e8f0;border-radius:10px;
            padding:18px 22px;margin-bottom:14px;border-left:4px solid #0891b2;'>
  <div style='font-size:12px;font-weight:700;color:#1e293b;margin-bottom:12px;
              letter-spacing:0.2px;'>
    &#128202; Tier-wise Anomaly Summary
  </div>"""

    for tier in ['T1','T2','T3']:
        td = tier_data_map.get(tier)
        if td is None: continue
        tc  = TIER_BG_MAP[tier]
        tl  = TIER_FULL_MAP[tier]
        otr_v = td['otr'];  otr_tc = metric_color(otr_v)
        mod_v = td['mod'];  mod_tc = metric_color(mod_v)

        if td['high'] == 0 and td['medium'] == 0:
            t_para = (
                f"All <b>{td['total_regions']} regions</b> in {tl} are operating "
                f"within SLA. OTR: <b style='color:{otr_tc};'>{otr_v}%</b> &nbsp;·&nbsp; "
                f"MOD: <b style='color:{mod_tc};'>{mod_v}%</b>. "
                f"No corrective action required for this tier."
            )
        else:
            high_list   = ", ".join(td['high_region_names'][:5])   or "—"
            medium_list = ", ".join(td['medium_region_names'][:5]) or "—"
            top_s  = td['top_stage']       or "N/A"
            top_sc = td['top_stage_count']
            ws     = td['worst_store']      or "—"

            t_para = (
                f"<b>{td['total_regions']} regions</b> &nbsp;·&nbsp; "
                f"<b style='color:#dc2626;'>{td['high']} HIGH</b> &nbsp;·&nbsp; "
                f"<b style='color:#ea580c;'>{td['medium']} MEDIUM</b> &nbsp;·&nbsp; "
                f"<b style='color:#16a34a;'>{td['ok']} OK</b>"
                f"&emsp;|&emsp;"
                f"OTR: <b style='color:{otr_tc};'>{otr_v}%</b> &nbsp;·&nbsp; "
                f"MOD: <b style='color:{mod_tc};'>{mod_v}%</b><br>"
            )
            if td['high_region_names']:
                t_para += (f"<span style='color:#dc2626;font-weight:700;'>■ HIGH regions:</span> "
                           f"{high_list}. ")
            if td['medium_region_names']:
                t_para += (f"<span style='color:#ea580c;font-weight:700;'>■ MEDIUM regions:</span> "
                           f"{medium_list}. ")
            if top_sc > 0:
                t_para += (f"<br>Most frequently breached stage: <b>{top_s}</b> "
                           f"({top_sc} of {td['total_regions']} regions breaching). ")
            if ws != "—":
                t_para += f"Worst performing store by InhouseSLA: <b>{ws}</b>."

        store_badge = (
            f"<span style='background:rgba(255,255,255,0.15);border-radius:12px;"
            f"padding:2px 10px;font-size:10px;'>"
            f"{td['high_stores']} HIGH stores &nbsp;·&nbsp; {td['medium_stores']} MEDIUM stores"
            f"</span>"
        )
        html += f"""
  <div style='border:1px solid #e2e8f0;border-radius:8px;margin-bottom:10px;overflow:hidden;'>
    <div style='background:{tc};color:#fff;padding:7px 14px;font-size:11px;font-weight:700;
                display:flex;align-items:center;gap:10px;flex-wrap:wrap;'>
      <span>{tl}</span>
      <span style='font-weight:400;opacity:0.75;font-size:10px;'>
        {td['total_regions']} regions &nbsp;·&nbsp;
        {td['total_stores']} stores &nbsp;·&nbsp;
        {td['raw_orders']:,} orders
      </span>
      <span style='margin-left:auto;'>{store_badge}</span>
    </div>
    <div style='padding:13px 18px;background:#fafbfc;'>
      <p style='font-size:12px;color:#334155;line-height:1.85;margin:0;'>{t_para}</p>
    </div>
  </div>"""

    html += "\n</div>\n"

    html += """
<div style='background:#fff;border:1px solid #e2e8f0;border-radius:10px;
            padding:18px 22px;margin-bottom:14px;border-left:4px solid #d97706;'>
  <div style='font-size:12px;font-weight:700;color:#1e293b;margin-bottom:12px;
              letter-spacing:0.2px;'>
    &#128269; Stage-wise Root Cause Analysis
  </div>
  <table style='width:100%;border-collapse:collapse;font-size:11px;
                border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;'>
    <thead>
      <tr style='background:#f8fafc;'>
        <th style='padding:9px 12px;text-align:left;color:#475569;font-weight:700;
                   border-bottom:2px solid #e2e8f0;min-width:110px;'>Stage</th>
        <th style='padding:9px 12px;text-align:center;color:#475569;font-weight:700;
                   border-bottom:2px solid #e2e8f0;min-width:80px;'>Hard Limit</th>
        <th style='padding:9px 12px;text-align:center;color:#475569;font-weight:700;
                   border-bottom:2px solid #e2e8f0;min-width:180px;'>Regions Breaching</th>
        <th style='padding:9px 12px;text-align:center;color:#475569;font-weight:700;
                   border-bottom:2px solid #e2e8f0;min-width:95px;'>Avg Excess</th>
        <th style='padding:9px 12px;text-align:left;color:#475569;font-weight:700;
                   border-bottom:2px solid #e2e8f0;'>Top Affected Regions</th>
      </tr>
    </thead>
    <tbody>"""

    for stage, info in sorted_stages:
        limit = SLA_CONFIG[stage]['hard_limit']
        bc    = info['count']
        pct   = info['pct']
        if bc > 0:
            sc  = "#dc2626" if pct>50 else ("#ea580c" if pct>25 else "#d97706")
            sbg = "#fef2f2" if pct>50 else ("#fff7ed" if pct>25 else "#fffbeb")
            bw  = min(pct,100)
            top_str = ", ".join(info['top']) if info['top'] else "—"
            html += f"""
      <tr style='background:{sbg};'>
        <td style='padding:9px 12px;font-weight:700;color:#1e293b;
                   border-bottom:1px solid #f1f5f9;border-left:3px solid {sc};'>{stage}</td>
        <td style='padding:9px 12px;text-align:center;border-bottom:1px solid #f1f5f9;
                   color:#64748b;'>{to_readable(limit)}</td>
        <td style='padding:9px 12px;text-align:center;border-bottom:1px solid #f1f5f9;'>
          <span style='font-weight:700;color:{sc};font-size:14px;'>{bc}</span>
          <span style='color:#94a3b8;font-size:10px;'>&nbsp;/ {total_regions} &nbsp;({pct}%)</span>
          <div style='display:inline-block;vertical-align:middle;margin-left:8px;
                      width:90px;height:5px;background:#e2e8f0;border-radius:3px;overflow:hidden;'>
            <div style='width:{bw}%;height:100%;background:{sc};'></div>
          </div>
        </td>
        <td style='padding:9px 12px;text-align:center;border-bottom:1px solid #f1f5f9;
                   font-weight:700;color:{sc};'>{to_readable(info['avg_excess'])}</td>
        <td style='padding:9px 12px;border-bottom:1px solid #f1f5f9;
                   color:#475569;font-size:10px;'>{top_str}</td>
      </tr>"""
        else:
            html += f"""
      <tr style='background:#f9fafb;'>
        <td style='padding:8px 12px;font-weight:600;color:#94a3b8;
                   border-bottom:1px solid #f1f5f9;border-left:3px solid #16a34a;'>{stage}</td>
        <td style='padding:8px 12px;text-align:center;border-bottom:1px solid #f1f5f9;
                   color:#94a3b8;font-size:10px;'>{to_readable(limit)}</td>
        <td style='padding:8px 12px;text-align:center;border-bottom:1px solid #f1f5f9;'>
          <span style='color:#16a34a;font-weight:700;'>✓ 0 breaches</span>
        </td>
        <td style='padding:8px 12px;text-align:center;border-bottom:1px solid #f1f5f9;
                   color:#16a34a;'>—</td>
        <td style='padding:8px 12px;border-bottom:1px solid #f1f5f9;
                   color:#94a3b8;font-size:10px;font-style:italic;'>Within SLA</td>
      </tr>"""

    html += """
    </tbody>
  </table>
</div>"""

    html += """
<div style='background:#fff;border:1px solid #e2e8f0;border-radius:10px;
            padding:18px 22px;margin-bottom:14px;border-left:4px solid #16a34a;'>
  <div style='font-size:12px;font-weight:700;color:#1e293b;margin-bottom:3px;
              letter-spacing:0.2px;'>
    &#9889; Recommended Actions
  </div>
  <div style='font-size:10px;color:#94a3b8;margin-bottom:14px;'>
    Sorted by breach severity &nbsp;·&nbsp; Prioritise HIGH stages first &nbsp;·&nbsp;
    All actions are guideline suggestions — validate with on-ground ops context before execution
  </div>"""

    shown = False
    for stage, info in sorted_stages:
        if info['count'] == 0: continue
        shown  = True
        am     = ACTION_MAP.get(stage, {})
        cause  = am.get('cause',  '—')
        impact = am.get('impact', '—')
        acts   = am.get('actions', [])
        pct    = info['pct']
        bc     = info['count']
        sc     = "#dc2626" if pct>50 else ("#ea580c" if pct>25 else "#d97706")
        sbg    = "#fef2f2" if pct>50 else ("#fff7ed" if pct>25 else "#fffbeb")
        slbl   = "CRITICAL" if pct>50 else ("HIGH" if pct>25 else "MODERATE")
        top_str= ", ".join(info['top']) if info['top'] else "—"
        acts_li= "".join(
            f"<li style='padding:3px 0;color:#334155;line-height:1.75;'>{a}</li>"
            for a in acts)
        html += f"""
  <div style='border:1px solid #e2e8f0;border-radius:8px;margin-bottom:12px;overflow:hidden;'>
    <div style='background:{sbg};padding:10px 16px;border-bottom:1px solid {sc}22;
                display:flex;align-items:center;gap:10px;flex-wrap:wrap;'>
      <span style='font-size:13px;font-weight:700;color:#1e293b;'>{stage}</span>
      <span style='background:{sc};color:#fff;font-size:9px;font-weight:700;
                   padding:2px 9px;border-radius:10px;letter-spacing:0.3px;'>{slbl}</span>
      <span style='margin-left:auto;font-size:10px;color:#64748b;'>
        {bc} of {total_regions} regions breaching &nbsp;·&nbsp;
        avg excess:&nbsp;<b style='color:{sc};'>{to_readable(info['avg_excess'])}</b>
      </span>
    </div>
    <div style='padding:14px 18px;display:grid;grid-template-columns:1fr 1fr;gap:16px;'>
      <div>
        <div style='font-size:9px;font-weight:700;color:#94a3b8;text-transform:uppercase;
                    letter-spacing:0.6px;margin-bottom:5px;'>Root Cause</div>
        <div style='font-size:11px;color:#334155;line-height:1.65;margin-bottom:10px;'>{cause}</div>
        <div style='font-size:9px;font-weight:700;color:#94a3b8;text-transform:uppercase;
                    letter-spacing:0.6px;margin-bottom:5px;'>Impact</div>
        <div style='font-size:11px;color:#334155;line-height:1.65;margin-bottom:10px;'>{impact}</div>
        <div style='font-size:9px;font-weight:700;color:#94a3b8;text-transform:uppercase;
                    letter-spacing:0.6px;margin-bottom:5px;'>Top Affected Regions</div>
        <div style='font-size:11px;font-weight:600;color:{sc};line-height:1.65;'>{top_str}</div>
      </div>
      <div>
        <div style='font-size:9px;font-weight:700;color:#94a3b8;text-transform:uppercase;
                    letter-spacing:0.6px;margin-bottom:8px;'>&#9654; Action Items</div>
        <ul style='margin:0;padding-left:16px;font-size:11px;list-style:disc;'>
          {acts_li}
        </ul>
      </div>
    </div>
  </div>"""

    if not shown:
        html += """
  <div style='padding:20px;text-align:center;color:#16a34a;font-size:13px;font-weight:700;
              background:#f0fdf4;border-radius:8px;border:1px solid #bbf7d0;'>
    ✓ No actions required — All stages are within SLA thresholds
  </div>"""

    html += """
</div>
<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
            padding:10px 16px;font-size:10px;color:#94a3b8;line-height:1.9;'>
  <b style='color:#475569;'>&#9432; Notes &amp; Methodology:</b>
  &nbsp; OTR% and MOD% use <b>all delivered orders</b> as denominator (no SLA Key filter).
  &nbsp; Stage timings use <b>SLA Key=Yes &amp; Delivered</b> orders only.
  &nbsp; <b>InhouseSLA</b> = avg(OP→INP) + avg(INP→PK) + avg(PK→BIN) per region.
  &nbsp; All stage values are <b>region-level averages</b>.
  &nbsp; <b>Avg Excess</b> = mean overshoot above hard limit across breaching regions.
</div>
</div>"""

    return html

# ═══════════════════════════════════════════════════════════════
#  BUILD FULL REPORT
# ═══════════════════════════════════════════════════════════════
def build_report(region_df, store_df, raw_df, all_delivered_df, filename):
    now    =datetime.now().strftime("%d %b %Y, %I:%M %p")
    total_r=len(region_df); total_s=len(store_df); total_o=len(raw_df)

    html=f"""<!DOCTYPE html>
<html><head><style>
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{font-family:'Segoe UI',system-ui,Arial,sans-serif;background:#f8fafc;
        color:#1e293b;font-size:12px;}}
  .wrap{{max-width:1200px;margin:auto;padding:14px;}}
  .hdr {{background:linear-gradient(135deg,#1e293b 0%,#334155 100%);color:#fff;
         border-radius:12px;padding:20px 26px;margin-bottom:18px;}}
  .hdr h1{{font-size:19px;font-weight:700;letter-spacing:0.3px;color:#fff;}}
  .hdr p {{font-size:11px;opacity:0.65;margin-top:3px;}}
  .sec-hdr{{display:flex;align-items:center;gap:10px;padding:8px 0;
            margin:18px 0 10px;border-bottom:2px solid #e2e8f0;}}
  .sec-hdr .title{{font-size:13px;font-weight:700;color:#1e293b;}}
  .sec-hdr .sub  {{font-size:11px;color:#94a3b8;}}
  .info{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
         padding:8px 12px;font-size:11px;color:#475569;
         margin-bottom:14px;line-height:1.8;}}
  .legend{{display:inline-flex;align-items:center;gap:8px;
           font-size:10px;color:#64748b;}}
  .lg{{width:11px;height:11px;border-radius:2px;display:inline-block;}}
  .ts{{text-align:right;font-size:10px;color:#94a3b8;margin-bottom:10px;}}
  details summary::-webkit-details-marker{{display:none;}}
</style></head>
<body><div class='wrap'>

<div class='ts'>&#128337; {now} &nbsp;·&nbsp; {filename}</div>
<div class='hdr'>
  <h1>&#128202; SLA Breach — RCA Report</h1>
  <p>Delivered · SLA Key=Yes ·
     {total_o:,} orders · {total_r} regions · {total_s} stores</p>
</div>

<div class='sec-hdr'>
  <span class='title'>&#128200; Performance Overview</span>
  <span class='sub'>Tier KPIs · Store counts · OTR% &amp; MOD%</span>
</div>
{tier_kpi_html(region_df, store_df, raw_df, all_delivered_df)}

<div class='info' style='margin-top:14px;'>
  <b>Column map:</b>&nbsp;
  OP to INP ← <code>Op to Inp 1</code> &nbsp;·&nbsp;
  INP to PK ← <code>Inp to Pk 1</code> &nbsp;·&nbsp;
  PK to BIN ← <code>Pk to Bin 1</code> &nbsp;·&nbsp;
  <b>InhouseSLA = avg(OP)+avg(INP)+avg(PK)</b> &nbsp;·&nbsp;
  BIN to RTS ← <code>Bin to Rts 1</code>
  <br>
  <b>OTR%:</b>&nbsp;count(OTR=10 &amp; delivered) ÷ count(all delivered)
  &nbsp;·&nbsp;
  <b>MOD%:</b>&nbsp;count(MOD=1 &amp; delivered) ÷ count(all delivered)
</div>

<div class='sec-hdr'>
  <span class='title'>&#128203; Summary Report</span>
  <span class='sub'>
    All regions · Tier header first · HIGH → MEDIUM → OK · InhouseSLA ↓
  </span>
</div>
{summary_report_html(region_df, raw_df, all_delivered_df)}

<div class='sec-hdr'>
  <span class='title'>&#128205; Regionwise Breakdown</span>
  <span class='sub'>
    Breaching only · OTR% with &gt;50%/≤50% · MOD% value · Click region → stores
  </span>
  <div class='legend' style='margin-left:auto;'>
    <span class='lg' style='background:#16a34a;'></span>OK &nbsp;&nbsp;
    <span class='lg' style='background:#dc2626;'></span>Breach
  </div>
</div>
{regionwise_breakdown_html(region_df, store_df, raw_df, all_delivered_df)}

<div class='sec-hdr'>
  <span class='title'>&#128203; Quick Summary Snapshot</span>
  <span class='sub'>
    Auto-generated narrative &nbsp;·&nbsp;
    Where are the anomalies &nbsp;·&nbsp;
    Root cause &nbsp;·&nbsp; Recommended actions
  </span>
</div>
{generate_summary_snapshot_html(region_df, store_df, raw_df, all_delivered_df)}

<br><br></div></body></html>"""
    return html

# ═══════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div style='background:linear-gradient(135deg,#1e293b,#334155);
            border-radius:12px;padding:22px 28px;color:#fff;
            font-family:Segoe UI,Arial,sans-serif;margin-bottom:24px;'>
  <h2 style='margin:0 0 6px;font-size:22px;letter-spacing:0.5px;color:#fff;'>
    📊 SLA Breach — RCA Report Generator
  </h2>
  <p style='margin:0;opacity:0.75;font-size:13px;'>
    Upload your Order Dump Excel file · Full report with Summary Snapshot + RCA generated automatically
  </p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📁 Upload Order Dump (Excel)",
    type=['xlsx','xls'],
    help="Sheet1 must contain: sa_name · OTR Status · SLA Key · OTR · MOD · stage cols"
)

if uploaded_file is not None:
    with st.spinner("⏳ Processing data and building report..."):
        try:
            file_bytes = uploaded_file.read()
            filename   = uploaded_file.name

            raw_df, all_delivered_df, before, after = load_dump(file_bytes)

            store_df  = aggregate_by_store(raw_df)
            store_df  = flag_breaches(store_df)
            region_df = aggregate_by_region(raw_df)
            region_df = flag_breaches(region_df)

            html_report = build_report(
                region_df, store_df, raw_df, all_delivered_df, filename
            )

            # ── Success banner ─────────────────────────────────────
            mod_yes = int((all_delivered_df['MOD']=='1').sum())
            st.success(
                f"✅ Report ready — "
                f"{len(raw_df):,} orders (SLA Key=Yes) · "
                f"{len(all_delivered_df):,} delivered (OTR%/MOD% base) · "
                f"{raw_df['Store'].nunique()} stores · "
                f"{raw_df['Region'].nunique()} regions · "
                f"MOD=1: {mod_yes:,}"
            )

            # ── Filter stats ───────────────────────────────────────
            st.info(
                f"🔍 Filter: {before:,} total rows → {after:,} kept "
                f"(Delivered + SLA Key=Yes) · {before-after:,} excluded"
            )

            # ── Download ───────────────────────────────────────────
            st.download_button(
                label="⬇️ Download Full Report as HTML",
                data=html_report.encode('utf-8'),
                file_name=f"SLA_RCA_{datetime.now().strftime('%d%b%Y_%H%M')}.html",
                mime="text/html",
                use_container_width=True
            )

            st.markdown("---")

            # ── Data Summary expander ──────────────────────────────
            with st.expander("📋 Data Summary — Store → Region Mapping", expanded=False):
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("All Delivered Orders", f"{len(all_delivered_df):,}")
                c2.metric("SLA Key=Yes Orders",   f"{len(raw_df):,}")
                c3.metric("Unique Stores",         f"{all_delivered_df['Store'].nunique():,}")
                c4.metric("Unique Regions",        f"{all_delivered_df['Region'].nunique()}")

                otr_yes = int((all_delivered_df['OTR']=='10').sum())
                c5,c6,_,_ = st.columns(4)
                c5.metric("MOD=1 (delivered)",  f"{mod_yes:,}")
                c6.metric("OTR=10 (delivered)", f"{otr_yes:,}")

                st.markdown("---")
                store_region_map = (
                    all_delivered_df[['Store','Region','Tier']]
                    .drop_duplicates()
                    .sort_values(['Tier','Region','Store'])
                    .reset_index(drop=True)
                )
                known   = store_region_map[store_region_map['Region']!='Unknown']
                unknown = store_region_map[store_region_map['Region']=='Unknown']

                tab_labels = ["T1 — Metro", "T2 — Major Cities", "T3 — Rural"]
                if not unknown.empty:
                    tab_labels.append(f"⚠️ Unmapped ({len(unknown)})")
                tabs = st.tabs(tab_labels)

                for i, tier in enumerate(['T1','T2','T3']):
                    with tabs[i]:
                        tier_data = known[known['Tier']==tier]
                        if tier_data.empty:
                            st.info("No stores in this tier.")
                            continue
                        for region in sorted(tier_data['Region'].unique()):
                            rr_stores = tier_data[tier_data['Region']==region]['Store'].tolist()
                            st.markdown(f"**{region}** — {len(rr_stores)} stores")
                            st.dataframe(
                                pd.DataFrame({'Store': rr_stores}),
                                use_container_width=True,
                                hide_index=True
                            )

                if not unknown.empty:
                    with tabs[-1]:
                        st.warning(
                            "These stores could not be mapped to a region. "
                            "Check store name format (expected: DS-T1BLR... or DS-BLR...)"
                        )
                        st.dataframe(unknown, use_container_width=True, hide_index=True)

            st.markdown("---")

            # ── Inline report ──────────────────────────────────────
            components.html(html_report, height=10000, scrolling=True)

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.exception(e)

else:
    st.info("👆 Upload your Excel dump file above to generate the report")
    st.markdown("""
    <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
                padding:16px 20px;font-size:12px;color:#475569;line-height:2.0;'>
      <b>What this report generates:</b><br>
      📈 <b>Performance Overview</b> — Tier KPI cards (Regions · Stores · Orders · OTR% · Store priorities · MOD%)<br>
      📋 <b>Summary Report</b> — All regions per tier: OTR% · Stage avg times · MOD% · Status<br>
      📍 <b>Regionwise Breakdown</b> — Breaching regions with OTR% store brackets + bar · Click → store drill-down<br>
      📄 <b>Quick Summary Snapshot</b> — Network narrative · Tier anomalies · Stage RCA table · Recommended actions<br>
      🔴 <b>HIGH</b> = InhouseSLA + BIN→RTS both breach &nbsp;·&nbsp;
      🟠 <b>MEDIUM</b> = Either one breaches &nbsp;·&nbsp;
      ✅ <b>OK</b> = All within SLA<br>
      <br>
      <b>Required columns in Sheet1:</b>&nbsp;
      <code>sa_name</code> · <code>OTR Status</code> · <code>SLA Key</code> ·
      <code>OTR</code> · <code>MOD</code> ·
      <code>Op to Inp 1</code> · <code>Inp to Pk 1</code> ·
      <code>Pk to Bin 1</code> · <code>Bin to Rts 1</code>
    </div>
    """, unsafe_allow_html=True)
