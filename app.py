# ═══════════════════════════════════════════════════════════════
# requirements.txt
#   streamlit
#   pandas
#   numpy
#   openpyxl
#   scikit-learn
#   openai>=1.0.0
#   google-generativeai
# ═══════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import io
import json
import streamlit.components.v1 as components

# ── Optional AI/ML imports (graceful fallback) ─────────────────
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    from openai import OpenAI as _OAI
    OPENAI_OK = True
except ImportError:
    OPENAI_OK = False

try:
    import google.generativeai as _genai
    GEMINI_OK = True
except ImportError:
    GEMINI_OK = False

st.set_page_config(
    page_title="SLA Breach — RCA + AI",
    page_icon="📊",
    layout="wide"
)

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════
SLA_CONFIG = {
    'InhouseSLA':   {'sla':180, 'buffer':30,  'hard_limit':210, 'avoidable':False},
    'OP to INP':    {'sla':20,  'buffer':5,   'hard_limit':25,  'avoidable':False},
    'INP to PK':    {'sla':120, 'buffer':30,  'hard_limit':150, 'avoidable':False},
    'PK to BIN':    {'sla':20,  'buffer':5,   'hard_limit':25,  'avoidable':False},
    'BIN to RTS':   {'sla':60,  'buffer':30,  'hard_limit':90,  'avoidable':False},
    'RTS to Reach': {'sla':600, 'buffer':300, 'hard_limit':900, 'avoidable':True},
    'RTS to DEL':   {'sla':600, 'buffer':300, 'hard_limit':900, 'avoidable':True},
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

# Region(1)+OTR%(1)+Stages(5)+MOD%(1)+Risk(1)+Status(1) = 10
TOTAL_COLS = 10

# AI: stage weights for risk scoring (must sum to 100)
RISK_WEIGHTS = {
    'InhouseSLA': 35,
    'BIN to RTS': 25,
    'INP to PK':  20,
    'OP to INP':  10,
    'PK to BIN':  10,
}

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
            if len(p)==3: return int(p[0])*3600+int(p[1])*60+int(float(p[2]))
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
        if isinstance(val,(int,float)): return str(int(val))
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
        s_ds=ds[ds['Store']==store]; t=len(s_ds)
        if t==0: continue
        pct=int((s_ds['OTR']=='10').sum()/t*100)
        if pct>50: above50+=1
        else: below_eq50+=1
    return above50+below_eq50, above50, below_eq50

# ═══════════════════════════════════════════════════════════════
#  SPLIT BAR + METRIC/STAGE CELL BUILDERS
# ═══════════════════════════════════════════════════════════════
def split_bar(wp, bp, wn=0, bn=0, width=120):
    if wp==0 and bp==0:
        return "<span style='color:#94a3b8;font-size:10px;'>—</span>"
    gp=max(0.0,min(100.0,wp)); rp=max(0.0,min(100.0,bp))
    def lbl(p): return f"{p:.0f}%" if p>=5 else ""
    gmin="2px" if gp>0 else "0"; rmin="2px" if rp>0 else "0"
    return f"""<div style='display:inline-flex;width:{width}px;height:20px;
                border-radius:4px;overflow:hidden;vertical-align:middle;
                border:1px solid #e2e8f0;'>
      <div style='width:{gp:.1f}%;min-width:{gmin};background:#16a34a;
                  display:flex;align-items:center;justify-content:center;'>
        <span style='color:#fff;font-size:9px;font-weight:700;
                     white-space:nowrap;padding:0 2px;'>{lbl(gp)}</span>
      </div>
      <div style='width:{rp:.1f}%;min-width:{rmin};background:#dc2626;
                  display:flex;align-items:center;justify-content:center;'>
        <span style='color:#fff;font-size:9px;font-weight:700;
                     white-space:nowrap;padding:0 2px;'>{lbl(rp)}</span>
      </div>
    </div>"""

def simple_metric_td(pct, pad="8px 10px"):
    col=metric_color(pct)
    bg="#fff5f5" if pct<50 else ("#fff8f0" if pct<70 else "#f0fdf4")
    return f"""<td style='padding:{pad};text-align:center;background:{bg};
               border-bottom:1px solid #f1f5f9;border-left:1px solid #f0f4f8;'>
      <span style='font-size:12px;font-weight:700;color:{col};'>{pct}%</span>
    </td>"""

def otr_region_td(pct, above50, below_eq50, total_s, pad="8px 8px", bar_width=85):
    col=metric_color(pct)
    bg="#fff5f5" if pct<50 else ("#fff8f0" if pct<70 else "#f0fdf4")
    bar_html=""
    if total_s>0:
        ab_pct=round(above50/total_s*100,1); be_pct=round(below_eq50/total_s*100,1)
        bar_html=f"""
        <div style='display:flex;gap:3px;justify-content:center;margin-top:3px;'>
          <span style='font-size:9px;color:#16a34a;font-weight:600;background:#f0fdf4;
                       padding:1px 4px;border-radius:3px;white-space:nowrap;'>
            &gt;50%:{above50}
          </span>
          <span style='font-size:9px;color:#dc2626;font-weight:600;background:#fef2f2;
                       padding:1px 4px;border-radius:3px;white-space:nowrap;'>
            ≤50%:{below_eq50}
          </span>
        </div>
        <div style='margin-top:3px;text-align:center;'>
          {split_bar(ab_pct,be_pct,above50,below_eq50,width=bar_width)}
        </div>"""
    return f"""<td style='padding:{pad};text-align:center;background:{bg};
               border-bottom:1px solid #f1f5f9;border-left:1px solid #f0f4f8;'>
      <span style='font-size:12px;font-weight:700;color:{col};'>{pct}%</span>
      {bar_html}
    </td>"""

def metric_th(label, sublabel, min_width="72px"):
    return f"""<th style='padding:9px 8px;text-align:center;min-width:{min_width};
               color:#475569;font-weight:600;border-bottom:2px solid #e2e8f0;'>
      <div style='font-size:11px;'>{label}</div>
      <div style='font-size:9px;color:#94a3b8;font-weight:400;margin-top:1px;'>
        {sublabel}
      </div>
    </th>"""

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
    if col not in stores_agg_df.columns or stores_agg_df.empty: return 0,0,0,0.0,0.0
    valid=stores_agg_df[stores_agg_df[col]>0]; total=len(valid)
    if total==0: return 0,0,0,0.0,0.0
    bc=int((valid[col]>limit).sum()); wc=total-bc
    return total,wc,bc,round(wc/total*100,1),round(bc/total*100,1)

def build_stage_cell_region(stage, avg, stores_agg_df, bar_width=110):
    limit=SLA_CONFIG[stage]['hard_limit']; is_breach=avg>limit and avg>0
    sev=severity_label(avg,limit) if is_breach else "OK"; sc=sev_color(sev)
    cell_bg="#fff5f5" if is_breach else "#fafafa"; wt="700" if is_breach else "500"
    total_,wn,bn,wp,bp=get_stage_pct_stores(stores_agg_df,stage)
    if total_>0:
        counts_html=f"""
        <div style='display:flex;gap:4px;justify-content:center;margin-top:3px;'>
          <span style='font-size:9px;color:#16a34a;font-weight:600;background:#f0fdf4;
                       padding:1px 5px;border-radius:3px;'>OK:{wn}</span>
          <span style='font-size:9px;color:#dc2626;font-weight:600;background:#fef2f2;
                       padding:1px 5px;border-radius:3px;'>B:{bn}</span>
        </div>
        <div style='margin-top:3px;text-align:center;'>
          {split_bar(wp,bp,wn,bn,width=bar_width)}
        </div>"""
    else:
        counts_html="<div style='font-size:9px;color:#cbd5e1;margin-top:3px;'>—</div>"
    return f"""<td style='padding:8px 7px;text-align:center;background:{cell_bg};
               border-bottom:1px solid #f1f5f9;border-left:1px solid #f0f4f8;'>
      <div style='font-size:12px;font-weight:{wt};color:{sc};'>{to_readable(avg)}</div>
      {counts_html}
    </td>"""

def build_stage_cell_store(stage, avg, raw_df_subset, bar_width=100):
    limit=SLA_CONFIG[stage]['hard_limit']; is_breach=avg>limit and avg>0
    sev=severity_label(avg,limit) if is_breach else "OK"; sc=sev_color(sev)
    cell_bg="#fff5f5" if is_breach else "#fafafa"; wt="700" if is_breach else "500"
    total_,wn,bn,wp,bp=get_stage_pct_orders(raw_df_subset,stage)
    if total_>0:
        counts_html=f"""
        <div style='display:flex;gap:4px;justify-content:center;margin-top:3px;'>
          <span style='font-size:9px;color:#16a34a;font-weight:600;background:#f0fdf4;
                       padding:1px 5px;border-radius:3px;'>OK:{wn}</span>
          <span style='font-size:9px;color:#dc2626;font-weight:600;background:#fef2f2;
                       padding:1px 5px;border-radius:3px;'>B:{bn}</span>
        </div>
        <div style='margin-top:3px;text-align:center;'>
          {split_bar(wp,bp,wn,bn,width=bar_width)}
        </div>"""
    else:
        counts_html="<div style='font-size:9px;color:#cbd5e1;margin-top:3px;'>—</div>"
    return f"""<td style='padding:8px 7px;text-align:center;background:{cell_bg};
               border-bottom:1px solid #f1f5f9;border-left:1px solid #f0f4f8;'>
      <div style='font-size:12px;font-weight:{wt};color:{sc};'>{to_readable(avg)}</div>
      {counts_html}
    </td>"""

# ═══════════════════════════════════════════════════════════════
#  LOAD & FILTER
# ═══════════════════════════════════════════════════════════════
def load_dump(fb):
    df=pd.read_excel(io.BytesIO(fb),sheet_name='Sheet1',header=0)
    df.columns=[str(c).strip() for c in df.columns]
    df=df.rename(columns={'sa_name':'Store'})
    df['OTR']=df['OTR'].apply(normalize_otr_val) if 'OTR' in df.columns else '0'
    df['MOD']=df['MOD'].apply(normalize_mod_val) if 'MOD' in df.columns else '0'
    df['_otr'] =df['OTR Status'].astype(str).str.strip().str.lower()
    df['_slak']=df['SLA Key'].astype(str).str.strip()

    all_del=df[df['_otr']=='delivered'].copy()
    all_del=all_del[all_del['Store'].notna()]
    all_del=all_del[all_del['Store'].astype(str).str.startswith('DS-')]
    all_del['Region']=all_del['Store'].apply(extract_region)
    all_del['Tier']  =all_del['Region'].apply(get_tier)
    all_delivered_df=all_del[['Store','OTR','MOD','Region','Tier']].reset_index(drop=True)

    df_main=df[(df['_otr']=='delivered')&(df['_slak']=='Yes')].copy()
    df_main.drop(columns=['_otr','_slak'],inplace=True)
    df_main=df_main.rename(columns=DUMP_COL_MAP)
    df_main=df_main[df_main['Store'].notna()]
    df_main=df_main[df_main['Store'].astype(str).str.startswith('DS-')]
    df_main=df_main.reset_index(drop=True)

    for col in STAGE_COLS:
        df_main[f'{col}_secs']=df_main[col].apply(time_to_seconds) if col in df_main.columns else 0
    df_main['InhouseSLA_secs']=(df_main['OP to INP_secs']+
                                 df_main['INP to PK_secs']+df_main['PK to BIN_secs'])
    df_main['Region']=df_main['Store'].apply(extract_region)
    df_main['Tier']  =df_main['Region'].apply(get_tier)
    return df_main, all_delivered_df

# ═══════════════════════════════════════════════════════════════
#  AGGREGATE + FLAG
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
                                 row.get('INP to PK_secs',0)+row.get('PK to BIN_secs',0))
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
                                 row.get('INP to PK_secs',0)+row.get('PK to BIN_secs',0))
        records.append(row)
    return pd.DataFrame(records)

def flag_breaches(df):
    for stage in list(SLA_CONFIG.keys()):
        col=f'{stage}_secs'; limit=SLA_CONFIG[stage]['hard_limit']
        df[f'{stage}_breach']=(df[col]>limit)&(df[col]>0) if col in df.columns else False
    return df

# ═══════════════════════════════════════════════════════════════
#  ██████████████████████████████████████████████████████████████
#  AI / ML MODULE
#  ──────────────────────────────────────────────────────────────
#  1. Isolation Forest  → anomaly_score (0-100), is_anomaly
#  2. Weighted Risk     → risk_score (0-100)
#  3. Rule-based RCA    → bottlenecks + recommendations (no API)
#  4. LLM RCA           → GPT-4o-mini or Gemini 1.5 Flash (API)
#  ██████████████████████████████████████████████████████████████
# ═══════════════════════════════════════════════════════════════

# ── 1. Isolation Forest Anomaly Detection ─────────────────────
def compute_anomaly_scores(store_df):
    """
    Uses Isolation Forest on store-level stage averages to detect
    statistically unusual stores (contamination = 10%).
    Adds: anomaly_score (0-100), is_anomaly (bool), anomaly_bonus (int).
    Falls back gracefully if scikit-learn is not installed.
    """
    store_df = store_df.copy()
    store_df['anomaly_score'] = 0.0
    store_df['is_anomaly']    = False
    store_df['anomaly_bonus'] = 0

    if not SKLEARN_OK or len(store_df) < 8:
        return store_df

    feat_cols = [c for c in
                 ['OP to INP_secs','INP to PK_secs','PK to BIN_secs',
                  'BIN to RTS_secs','InhouseSLA_secs']
                 if c in store_df.columns]
    if not feat_cols:
        return store_df

    X        = store_df[feat_cols].fillna(0).values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso    = IsolationForest(contamination=0.10, random_state=42, n_estimators=150)
    iso.fit(X_scaled)
    scores = iso.decision_function(X_scaled)   # more negative = more anomalous
    labels = iso.predict(X_scaled)             # -1 anomaly | +1 normal

    # Normalise to 0-100 (100 = most anomalous)
    mn, mx = scores.min(), scores.max()
    norm   = 1.0 - (scores - mn) / (mx - mn) if mx > mn else np.zeros(len(scores))

    store_df['anomaly_score'] = np.round(norm * 100, 1)
    store_df['is_anomaly']    = (labels == -1)
    # Anomaly bonus lifts risk score by up to +15 pts
    store_df['anomaly_bonus'] = store_df['is_anomaly'].apply(lambda x: 15 if x else 0)
    return store_df


# ── 2. Weighted Risk Score ─────────────────────────────────────
def compute_risk_score(row):
    """
    0-100 score per store / region row.
    Based on: breach severity per stage (weighted) + multi-breach
    penalty + anomaly detection bonus.
    """
    score        = 0.0
    breach_count = 0
    for stage, weight in RISK_WEIGHTS.items():
        col   = f'{stage}_secs'
        limit = SLA_CONFIG[stage]['hard_limit']
        val   = row.get(col, 0)
        if val > limit and val > 0:
            breach_count += 1
            excess_ratio  = min(3.0, (val - limit) / limit)
            score        += weight * (excess_ratio / 3.0)
    if breach_count >= 2:
        score = min(95, score * 1.2)           # multi-breach penalty
    score += row.get('anomaly_bonus', 0)       # anomaly lift
    return min(100, round(score))

def add_risk_scores(df):
    df = df.copy()
    if 'anomaly_bonus' not in df.columns:
        df['anomaly_bonus'] = 0
    df['risk_score'] = df.apply(compute_risk_score, axis=1)
    return df

def risk_meta(score):
    """Returns (label, color, bg) for a given risk score."""
    if score >= 70: return "CRITICAL", "#dc2626", "#fef2f2"
    if score >= 45: return "HIGH",     "#ea580c", "#fff7ed"
    if score >= 20: return "MEDIUM",   "#d97706", "#fffbeb"
    return                 "LOW",      "#16a34a", "#f0fdf4"

def risk_badge_html(score):
    """Compact risk badge with mini progress bar — used in tables."""
    lbl, col, bg = risk_meta(score)
    bar = max(3, score)
    return f"""
    <div style='text-align:center;'>
      <div style='font-size:13px;font-weight:800;color:{col};line-height:1.1;'>
        {score}
      </div>
      <div style='font-size:8px;font-weight:700;color:{col};background:{bg};
                  border-radius:3px;padding:1px 4px;display:inline-block;
                  margin-top:1px;'>{lbl}</div>
      <div style='width:44px;height:4px;background:#e2e8f0;border-radius:2px;
                  margin:3px auto 0;'>
        <div style='width:{bar}%;height:100%;background:{col};
                    border-radius:2px;'></div>
      </div>
    </div>"""

def anomaly_flag_html(is_anomaly):
    if not is_anomaly: return ""
    return """<span style='font-size:9px;background:#fef3c7;color:#92400e;
                           border:1px solid #fcd34d;border-radius:3px;
                           padding:1px 4px;font-weight:700;margin-left:4px;'>
                ⚠ ANOMALY
              </span>"""


# ── 3. Rule-Based RCA (no API needed) ─────────────────────────
def generate_rule_based_rca(region_name, region_row, tier_label,
                             region_otr, region_mod, risk_score):
    """
    Deterministic RCA using domain knowledge about dark-store ops.
    Returns a structured dict identical in shape to LLM output.
    """
    def breach(stage):
        col   = f'{stage}_secs'
        limit = SLA_CONFIG[stage]['hard_limit']
        val   = region_row.get(col, 0)
        return val > limit and val > 0, val, limit

    roots = []; recs = []; primary = None

    ib, iv, il = breach('INP to PK')
    if ib:
        excess = round((iv-il)/il*100)
        roots.append(f"INP→PK is {excess}% over limit ({to_readable(iv)} vs {to_readable(il)}) — picking speed is the primary constraint")
        recs.append("Review picker allocation & wave planning for INP→PK stage")
        recs.append("Audit putaway accuracy — misplaced SKUs directly slow picking")
        primary = primary or "INP → PK (Picking)"

    ob, ov, ol = breach('OP to INP')
    if ob:
        excess = round((ov-ol)/ol*100)
        roots.append(f"OP→INP is {excess}% over limit — order acceptance is delayed")
        recs.append("Check order acceptance automation — OP→INP lag signals manual intervention or system latency")
        primary = primary or "OP → INP (Acceptance)"

    pb, pv, pl = breach('PK to BIN')
    if pb:
        roots.append(f"PK→BIN is over limit ({to_readable(pv)}) — binning is a bottleneck")
        recs.append("Optimise bin assignment logic and review bin capacity utilisation")
        primary = primary or "PK → BIN (Binning)"

    bb, bv, bl = breach('BIN to RTS')
    if bb:
        excess = round((bv-bl)/bl*100)
        roots.append(f"BIN→RTS is {excess}% over limit — dispatch handover is slow")
        recs.append("Coordinate with last-mile partner on RTS pickup SLA adherence")
        recs.append("Enable real-time RTS alerts to reduce rider wait time at store")
        primary = primary or "BIN → RTS (Dispatch)"

    if region_otr < 50:
        roots.append(f"OTR% is critically low at {region_otr}% — majority of orders are delayed")
        recs.append("Deep-dive OTR failures by store and slot to isolate peak-hour issues")

    if region_mod > 30:
        roots.append(f"MOD% is elevated at {region_mod}% — order modifications are disrupting fulfilment flow")
        recs.append("Improve inventory accuracy and real-time stock sync to reduce MODs")

    if not roots:
        roots.append("No major stage-level bottlenecks detected at region average — review store-level outliers")
        recs.append("Monitor store-level anomaly flags for early warning signals")

    urgency = ("IMMEDIATE" if risk_score >= 70 else
               "24H"       if risk_score >= 45 else
               "48H"       if risk_score >= 20 else "MONITOR")

    return {
        "summary": (
            f"{region_name} ({tier_label}) is showing "
            f"{'critical' if risk_score>=70 else 'elevated'} SLA breach patterns "
            f"with a risk score of {risk_score}/100. "
            f"The primary bottleneck is {primary or 'multiple stages'}, "
            f"with OTR at {region_otr}% and MOD at {region_mod}%."
        ),
        "root_causes":       roots[:4],
        "bottleneck_stage":  primary or "See stage details",
        "recommendations":   recs[:4],
        "predicted_trend":   "STABLE",
        "predicted_trend_reason": "Single-snapshot analysis — upload multiple dumps to detect trend",
        "urgency":           urgency,
        "risk_score":        risk_score,
        "_source":           "rule_based",
    }


# ── 4. LLM RCA (OpenAI / Gemini) ──────────────────────────────
def _build_llm_prompt(region_name, region_row, tier_label,
                       region_otr, region_mod, risk_score, stores_data):
    def fmt(stage):
        col   = f'{stage}_secs'
        limit = SLA_CONFIG[stage]['hard_limit']
        val   = region_row.get(col, 0)
        status= '⚠ BREACH' if val>limit and val>0 else '✅ OK'
        return f"{to_readable(val)} / {to_readable(limit)} [{status}]"

    store_lines = "\n".join([
        f"  • {s['name']}: Risk={s['risk']}/100, IH-SLA={s['ih']}, "
        f"BIN→RTS={s['br']}, Orders={s['orders']}, Anomaly={s['anomaly']}"
        for s in stores_data[:5]
    ]) if stores_data else "  (no store-level data)"

    return f"""You are a senior operations analyst for a quick-commerce dark-store network.

Analyse this SLA breach data and return ONLY valid JSON — no markdown.

REGION: {region_name} ({tier_label})
PRIORITY: {region_row.get('_priority','OK')} | RISK SCORE: {risk_score}/100

STAGE PERFORMANCE (avg / limit):
  OP → INP   : {fmt('OP to INP')}
  INP → PK   : {fmt('INP to PK')}
  PK → BIN   : {fmt('PK to BIN')}
  InhouseSLA : {fmt('InhouseSLA')}  ← OP+INP+PK combined
  BIN → RTS  : {fmt('BIN to RTS')}

METRICS:
  OTR%: {region_otr}%  (On-Time Rate, target ≥70%)
  MOD%: {region_mod}%  (Modification Rate, lower=better)

TOP BREACHING STORES:
{store_lines}

STAGE DEFINITIONS:
  OP→INP : Order placed → picked (SLA 25 s)
  INP→PK : Picked → packing (SLA 2.5 min)
  PK→BIN : Packing → bin (SLA 25 s)
  IH-SLA : Total in-store (SLA 3.5 min)
  BIN→RTS: Bin → ready to ship (SLA 1.5 min)

Return this EXACT JSON structure:
{{
  "summary": "2-3 sentence executive summary",
  "root_causes": ["cause 1", "cause 2", "cause 3"],
  "bottleneck_stage": "single most critical stage name",
  "recommendations": ["action 1", "action 2", "action 3", "action 4"],
  "predicted_trend": "IMPROVING or STABLE or WORSENING",
  "predicted_trend_reason": "one-line reason",
  "urgency": "IMMEDIATE or 24H or 48H or MONITOR"
}}"""

def _call_openai(prompt, api_key, model):
    try:
        client = _OAI(api_key=api_key)
        resp   = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":"You are an expert operations analyst. Return valid JSON only."},
                {"role":"user","content":prompt}
            ],
            temperature=0.25, max_tokens=700,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"): raw = raw.split("```")[1].lstrip("json").strip()
        return json.loads(raw)
    except Exception as e:
        return {"_error": str(e)}

def _call_gemini(prompt, api_key):
    try:
        _genai.configure(api_key=api_key)
        model = _genai.GenerativeModel('gemini-1.5-flash')
        resp  = model.generate_content(
            prompt,
            generation_config=_genai.types.GenerationConfig(temperature=0.25)
        )
        raw = resp.text.strip()
        if raw.startswith("```"): raw = raw.split("```")[1].lstrip("json").strip()
        return json.loads(raw)
    except Exception as e:
        return {"_error": str(e)}

def generate_llm_rca(region_name, region_row, tier_label,
                      region_otr, region_mod, risk_score,
                      stores_data, provider, api_key, model):
    """Calls chosen LLM provider; falls back to rule-based on error."""
    prompt = _build_llm_prompt(region_name, region_row, tier_label,
                                region_otr, region_mod, risk_score, stores_data)
    if provider == "OpenAI" and OPENAI_OK:
        result = _call_openai(prompt, api_key, model)
    elif provider == "Gemini" and GEMINI_OK:
        result = _call_gemini(prompt, api_key)
    else:
        result = {"_error": "provider_unavailable"}

    if "_error" in result:
        # fallback
        result = generate_rule_based_rca(
            region_name, region_row, tier_label,
            region_otr, region_mod, risk_score)
    else:
        result['risk_score'] = risk_score
        result['_source']    = 'llm'
    return result


# ── RCA Card HTML ──────────────────────────────────────────────
def render_rca_card_html(rca, region_name):
    risk  = rca.get('risk_score', 0)
    lbl, col, bg = risk_meta(risk)
    is_llm = rca.get('_source') == 'llm'

    trend       = rca.get('predicted_trend', 'STABLE')
    trend_icons = {'IMPROVING':'📈','STABLE':'➡️','WORSENING':'📉'}
    trend_cols  = {'IMPROVING':'#16a34a','STABLE':'#d97706','WORSENING':'#dc2626'}
    tc          = trend_cols.get(trend, '#64748b')
    urgency     = rca.get('urgency','MONITOR')
    urg_col     = {'IMMEDIATE':'#dc2626','24H':'#ea580c','48H':'#d97706',
                   'MONITOR':'#16a34a'}.get(urgency,'#64748b')

    causes_html = "".join(
        f"""<div style='display:flex;gap:6px;margin-bottom:5px;align-items:flex-start;'>
              <span style='color:#ea580c;font-size:11px;min-width:14px;'>⚠</span>
              <span style='font-size:11px;color:#374151;line-height:1.5;'>{c}</span>
            </div>"""
        for c in rca.get('root_causes', []))

    recs_html = "".join(
        f"""<div style='display:flex;gap:6px;margin-bottom:5px;align-items:flex-start;'>
              <span style='color:#16a34a;font-size:11px;min-width:14px;'>→</span>
              <span style='font-size:11px;color:#374151;line-height:1.5;'>{r}</span>
            </div>"""
        for r in rca.get('recommendations', []))

    source_tag = (
        "<span style='font-size:9px;color:#7c3aed;font-weight:700;background:#ede9fe;"
        "padding:1px 6px;border-radius:3px;'>🤖 AI Generated</span>"
        if is_llm else
        "<span style='font-size:9px;color:#475569;font-weight:600;background:#f1f5f9;"
        "padding:1px 6px;border-radius:3px;'>📐 Rule-Based</span>"
    )

    bottleneck = rca.get('bottleneck_stage','')

    return f"""
    <div style='background:#fff;border:1px solid #e2e8f0;border-radius:8px;
                padding:14px 16px;margin-bottom:12px;border-left:4px solid {col};'>
      <!-- header -->
      <div style='display:flex;justify-content:space-between;align-items:center;
                  flex-wrap:wrap;gap:6px;margin-bottom:10px;'>
        <div style='font-size:13px;font-weight:700;color:#1e293b;'>📍 {region_name}</div>
        <div style='display:flex;gap:6px;align-items:center;flex-wrap:wrap;'>
          {source_tag}
          <span style='font-size:10px;font-weight:700;color:{col};background:{bg};
                       border:1px solid {col};border-radius:4px;padding:2px 8px;'>
            Risk {risk}/100 — {lbl}
          </span>
          <span style='font-size:10px;font-weight:700;color:{tc};background:#f8fafc;
                       border:1px solid {tc};border-radius:4px;padding:2px 8px;'>
            {trend_icons.get(trend,'➡️')} {trend}
          </span>
          <span style='font-size:10px;font-weight:700;color:{urg_col};background:#f8fafc;
                       border:1px solid {urg_col};border-radius:4px;padding:2px 8px;'>
            ⏱ {urgency}
          </span>
        </div>
      </div>
      <!-- summary -->
      <div style='background:#f8fafc;border-left:3px solid {col};border-radius:0 4px 4px 0;
                  padding:8px 10px;margin-bottom:10px;font-size:11px;color:#374151;
                  line-height:1.6;font-style:italic;'>
        {rca.get('summary','')}
      </div>
      <!-- bottleneck -->
      {'<div style="background:#fff3ed;border-left:3px solid #ea580c;padding:6px 10px;border-radius:0 4px 4px 0;margin-bottom:10px;"><span style="font-size:10px;font-weight:700;color:#ea580c;">🎯 Primary Bottleneck: </span><span style="font-size:11px;color:#374151;">' + bottleneck + '</span></div>' if bottleneck else ''}
      <!-- trend reason -->
      {'<div style="font-size:10px;color:#94a3b8;font-style:italic;margin-bottom:10px;">Trend reason: ' + rca.get("predicted_trend_reason","") + '</div>' if rca.get("predicted_trend_reason") else ''}
      <!-- 2-col: causes + recs -->
      <div style='display:grid;grid-template-columns:1fr 1fr;gap:14px;'>
        <div>
          <div style='font-size:10px;font-weight:700;color:#475569;
                      text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;'>
            🔍 Root Causes
          </div>
          {causes_html}
        </div>
        <div>
          <div style='font-size:10px;font-weight:700;color:#475569;
                      text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;'>
            💡 Recommendations
          </div>
          {recs_html}
        </div>
      </div>
    </div>"""


# ── AI Section HTML (embedded in main report) ──────────────────
def ai_section_html(store_df, all_delivered_df, rca_map):
    """
    Renders the full AI Insights block:
    - Summary badges
    - Top-10 highest risk stores table
    - RCA cards for analysed regions
    """
    if 'risk_score' not in store_df.columns:
        return ""

    n_anomalies = int(store_df.get('is_anomaly', pd.Series(dtype=bool)).sum()) \
                  if 'is_anomaly' in store_df.columns else 0
    n_critical  = int((store_df['risk_score'] >= 70).sum())
    n_high      = int(((store_df['risk_score'] >= 45) & (store_df['risk_score'] < 70)).sum())

    top10 = store_df.nlargest(10, 'risk_score')
    rows  = ""
    for _, sr in top10.iterrows():
        rs       = int(sr.get('risk_score', 0))
        is_anom  = bool(sr.get('is_anomaly', False))
        anom_flag= anomaly_flag_html(is_anom)
        otr      = compute_otr_pct(all_delivered_df, store=sr['Store'])
        mod      = compute_mod_pct(all_delivered_df, store=sr['Store'])
        ih_col   = "#dc2626" if sr.get('InhouseSLA_breach') else "#16a34a"
        br_col   = "#dc2626" if sr.get('BIN to RTS_breach') else "#16a34a"
        rows += f"""
        <tr>
          <td style='padding:7px 10px;font-size:10px;font-weight:600;color:#1e293b;
                     border-bottom:1px solid #f1f5f9;white-space:nowrap;'>
            {sr['Store']}{anom_flag}
            <div style='font-size:9px;color:#94a3b8;font-weight:400;'>
              {sr.get('Region','')} · {sr.get('Tier','')}
            </div>
          </td>
          <td style='padding:7px 8px;text-align:center;border-bottom:1px solid #f1f5f9;'>
            {risk_badge_html(rs)}
          </td>
          <td style='padding:7px 8px;text-align:center;border-bottom:1px solid #f1f5f9;
                     font-size:11px;font-weight:700;color:{ih_col};'>
            {to_readable(sr.get('InhouseSLA_secs',0))}
          </td>
          <td style='padding:7px 8px;text-align:center;border-bottom:1px solid #f1f5f9;
                     font-size:11px;font-weight:700;color:{br_col};'>
            {to_readable(sr.get('BIN to RTS_secs',0))}
          </td>
          <td style='padding:7px 8px;text-align:center;border-bottom:1px solid #f1f5f9;'>
            <span style='font-size:11px;font-weight:700;color:{metric_color(otr)};'>
              {otr}%
            </span>
          </td>
          <td style='padding:7px 8px;text-align:center;border-bottom:1px solid #f1f5f9;'>
            <span style='font-size:11px;font-weight:700;color:{metric_color(mod)};'>
              {mod}%
            </span>
          </td>
          <td style='padding:7px 8px;text-align:center;border-bottom:1px solid #f1f5f9;
                     font-size:11px;color:#64748b;'>
            {sr.get('anomaly_score',0):.0f}
          </td>
        </tr>"""

    rca_cards = "".join(
        render_rca_card_html(rca, rname)
        for rname, rca in rca_map.items()
    )

    no_llm_tip = (
        "<div style='font-size:10px;color:#94a3b8;font-style:italic;margin-top:8px;'>"
        "💡 Add an OpenAI or Gemini API key in the sidebar to generate AI-powered RCA "
        "narratives for the top breaching regions.</div>"
        if not rca_map else ""
    )

    return f"""
    <div style='background:#fff;border:1px solid #e2e8f0;border-radius:10px;
                padding:16px;margin-bottom:16px;'>

      <!-- Section header -->
      <div style='display:flex;align-items:center;gap:10px;margin-bottom:14px;
                  flex-wrap:wrap;'>
        <div style='font-size:20px;'>🤖</div>
        <div>
          <div style='font-size:13px;font-weight:700;color:#1e293b;'>
            AI Anomaly Detection &amp; Risk Scoring
          </div>
          <div style='font-size:10px;color:#94a3b8;margin-top:2px;'>
            Isolation Forest (sklearn) · Weighted Risk Model ·
            {'<span style="color:#7c3aed;font-weight:600;">LLM RCA Active</span>'
             if rca_map else 'Rule-Based RCA'}
          </div>
        </div>
        <div style='margin-left:auto;display:flex;gap:6px;flex-wrap:wrap;'>
          <span style='font-size:10px;background:#fef2f2;color:#dc2626;font-weight:700;
                       padding:3px 9px;border-radius:4px;border:1px solid #fecaca;'>
            🔴 {n_critical} CRITICAL
          </span>
          <span style='font-size:10px;background:#fff7ed;color:#ea580c;font-weight:700;
                       padding:3px 9px;border-radius:4px;border:1px solid #fed7aa;'>
            🟠 {n_high} HIGH
          </span>
          <span style='font-size:10px;background:#fef3c7;color:#92400e;font-weight:700;
                       padding:3px 9px;border-radius:4px;border:1px solid #fcd34d;'>
            ⚠ {n_anomalies} Anomalies
          </span>
        </div>
      </div>

      <!-- Top risk stores table -->
      <div style='font-size:11px;font-weight:700;color:#475569;margin-bottom:8px;'>
        🎯 Top 10 Highest Risk Stores
      </div>
      <div style='overflow-x:auto;'>
      <table style='width:100%;border-collapse:collapse;font-size:11px;background:#fff;
                    border-radius:8px;overflow:hidden;border:1px solid #e2e8f0;
                    margin-bottom:16px;'>
        <thead>
          <tr style='background:#f1f5f9;'>
            <th style='padding:8px 10px;text-align:left;color:#475569;font-weight:700;
                       border-bottom:2px solid #e2e8f0;min-width:210px;'>Store</th>
            <th style='padding:8px 8px;text-align:center;color:#475569;font-weight:700;
                       border-bottom:2px solid #e2e8f0;min-width:80px;'>Risk Score</th>
            <th style='padding:8px 8px;text-align:center;color:#475569;font-weight:700;
                       border-bottom:2px solid #e2e8f0;min-width:90px;'>InhouseSLA</th>
            <th style='padding:8px 8px;text-align:center;color:#475569;font-weight:700;
                       border-bottom:2px solid #e2e8f0;min-width:90px;'>BIN→RTS</th>
            <th style='padding:8px 8px;text-align:center;color:#475569;font-weight:700;
                       border-bottom:2px solid #e2e8f0;min-width:60px;'>OTR%</th>
            <th style='padding:8px 8px;text-align:center;color:#475569;font-weight:700;
                       border-bottom:2px solid #e2e8f0;min-width:60px;'>MOD%</th>
            <th style='padding:8px 8px;text-align:center;color:#475569;font-weight:700;
                       border-bottom:2px solid #e2e8f0;min-width:70px;'>Anomaly<br>Score</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
      </div>

      <!-- RCA cards -->
      {'<div style="font-size:11px;font-weight:700;color:#475569;margin-bottom:10px;">📋 Root Cause Analysis — Breaching Regions</div>' + rca_cards if rca_cards else ''}
      {no_llm_tip}
    </div>"""


# ═══════════════════════════════════════════════════════════════
#  TIER KPI CARDS (unchanged logic, risk counts added)
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

        s_high=0;s_medium=0;s_ok=0;s_ih=0;s_br=0;s_crit=0
        for _,sr in t_stores.iterrows():
            p=get_priority(sr)
            if p=='HIGH':     s_high  +=1
            elif p=='MEDIUM': s_medium+=1
            else:             s_ok    +=1
            if sr.get('InhouseSLA_breach',False): s_ih+=1
            if sr.get('BIN to RTS_breach',False): s_br+=1
            if sr.get('risk_score',0) >= 70:      s_crit+=1

        tier_otr=compute_otr_pct(all_delivered_df,tier=tier)
        tier_mod=compute_mod_pct(all_delivered_df,tier=tier)
        otr_col=metric_color(tier_otr); mod_col=metric_color(tier_mod)
        meta=TIER_META[tier]

        def std_card(val,lbl,col,bl=True):
            border="border-left:1px solid #f1f5f9;" if bl else ""
            return (f"<div style='flex:1;padding:12px 8px;text-align:center;{border}'>"
                    f"<div style='font-size:20px;font-weight:700;color:{col};'>{val}</div>"
                    f"<div style='font-size:9px;color:#94a3b8;margin-top:3px;"
                    f"white-space:nowrap;'>{lbl}</div></div>")

        html+=f"""
        <div style='margin-bottom:14px;'>
          <div style='background:{meta["color"]};color:#fff;border-radius:8px 8px 0 0;
                      padding:7px 16px;font-size:11px;font-weight:700;'>
            {meta["label"]} &nbsp;·&nbsp;
            <span style='opacity:0.75;font-weight:400;'>
              {len(t_regions)} regions &nbsp;·&nbsp;
              {len(t_stores)} stores &nbsp;·&nbsp; {len(t_raw):,} orders
            </span>
          </div>
          <div style='display:flex;background:#fff;border:1px solid #e2e8f0;
                      border-top:none;border-radius:0 0 8px 8px;overflow:hidden;'>
            {std_card(len(t_regions),"Regions","#1e293b",False)}
            {std_card(len(t_stores),"Stores","#334155")}
            {std_card(f"{len(t_raw):,}","Orders","#475569")}
            <div style='flex:1;padding:12px 8px;text-align:center;border-left:1px solid #f1f5f9;'>
              <div style='font-size:20px;font-weight:700;color:{otr_col};'>{tier_otr}%</div>
              <div style='font-size:9px;color:#94a3b8;margin-top:3px;'>OTR %</div>
            </div>
            {std_card(s_high,"HIGH Stores","#dc2626")}
            {std_card(s_medium,"MEDIUM Stores","#ea580c")}
            {std_card(s_ok,"Stores OK","#16a34a")}
            {std_card(s_ih,"IH-SLA Breach","#ea580c")}
            {std_card(s_br,"BIN→RTS Breach","#d97706")}
            {std_card(s_crit,"AI Critical🤖","#dc2626")}
            <div style='flex:1;padding:12px 8px;text-align:center;border-left:1px solid #f1f5f9;'>
              <div style='font-size:20px;font-weight:700;color:{mod_col};'>{tier_mod}%</div>
              <div style='font-size:9px;color:#94a3b8;margin-top:3px;'>MOD %</div>
            </div>
          </div>
        </div>"""
    return html


# ═══════════════════════════════════════════════════════════════
#  SUMMARY REPORT  (+ Risk column)
# ═══════════════════════════════════════════════════════════════
def summary_report_html(region_df, raw_df, all_delivered_df):
    TIER_LABELS={'T1':'Tier 1 — Metro','T2':'Tier 2 — Major Cities','T3':'Tier 3 — Rural'}
    TIER_BG={'T1':'#1e293b','T2':'#1e3a5f','T3':'#1e4a3f'}
    STAGE_SHORT={'OP to INP':'OP→INP','INP to PK':'INP→PK','PK to BIN':'PK→BIN',
                 'InhouseSLA':'InhouseSLA','BIN to RTS':'BIN→RTS'}

    stage_th="".join(f"""
        <th style='padding:9px 10px;text-align:center;min-width:105px;color:#475569;
                   font-weight:600;border-bottom:2px solid #e2e8f0;'>
          <div style='font-size:11px;'>{STAGE_SHORT[s]}</div>
          <div style='font-size:9px;color:#94a3b8;font-weight:400;margin-top:2px;'>
            Limit:{to_readable(SLA_CONFIG[s]['hard_limit'])}
          </div>
        </th>""" for s in SUMMARY_STAGES)

    html=f"""
    <div style='overflow-x:auto;'>
    <table style='width:100%;border-collapse:collapse;font-size:11px;background:#fff;
                  border-radius:10px;overflow:hidden;border:1px solid #e2e8f0;'>
      <thead>
        <tr style='background:#f1f5f9;'>
          <th style='padding:9px 14px;text-align:left;color:#475569;font-weight:600;
                     border-bottom:2px solid #e2e8f0;min-width:180px;'>
            Region
            <div style='font-size:9px;color:#94a3b8;font-weight:400;margin-top:2px;'>
              HIGH → MEDIUM → OK · InhouseSLA ↓
            </div>
          </th>
          {metric_th("OTR %","OTR=10","65px")}
          {stage_th}
          {metric_th("MOD %","MOD=1","65px")}
          {metric_th("Risk 🤖","0-100","68px")}
          <th style='padding:9px 10px;text-align:center;border-bottom:2px solid #e2e8f0;
                     min-width:78px;color:#475569;font-weight:600;'>Status</th>
        </tr>
      </thead>
      <tbody>"""

    for tier in ['T1','T2','T3']:
        t_regions=region_df[region_df['Tier']==tier].copy()
        if t_regions.empty: continue
        t_regions['_priority']=t_regions.apply(get_priority,axis=1)
        t_regions['_psort']   =t_regions['_priority'].map(priority_sort_key)
        t_regions=t_regions.sort_values(['_psort','InhouseSLA_secs'],
                                         ascending=[True,False]).reset_index(drop=True)
        n_high=int((t_regions['_priority']=='HIGH').sum())
        n_med =int((t_regions['_priority']=='MEDIUM').sum())
        n_ok  =int((t_regions['_priority']=='OK').sum())
        tier_otr=compute_otr_pct(all_delivered_df,tier=tier)
        tier_mod=compute_mod_pct(all_delivered_df,tier=tier)

        html+=f"""
      <tr>
        <td colspan='{TOTAL_COLS}'
            style='background:{TIER_BG[tier]};color:#fff;font-size:11px;
                   font-weight:700;padding:7px 14px;letter-spacing:0.3px;'>
          {TIER_LABELS[tier]} &nbsp;·&nbsp; {len(t_regions)} regions
          &nbsp;&nbsp;
          <span style='color:#fca5a5;font-size:10px;'>■ {n_high} HIGH</span>
          &nbsp;<span style='color:#fdba74;font-size:10px;'>■ {n_med} MEDIUM</span>
          &nbsp;<span style='color:#86efac;font-size:10px;'>■ {n_ok} OK</span>
          &nbsp;&nbsp;
          <span style='color:#a5f3fc;font-size:10px;'>OTR:{tier_otr}%</span>
          &nbsp;<span style='color:#c4b5fd;font-size:10px;'>MOD:{tier_mod}%</span>
        </td>
      </tr>"""

        for _,row in t_regions.iterrows():
            region_name=row['Region']; priority=row['_priority']
            r_raw=raw_df[raw_df['Region']==region_name]
            region_otr=compute_otr_pct(all_delivered_df,region=region_name)
            region_mod=compute_mod_pct(all_delivered_df,region=region_name)
            risk_val=int(row.get('risk_score',0))
            p_styles={"HIGH":("#dc2626","#fef2f2","#fecaca"),
                      "MEDIUM":("#ea580c","#fff7ed","#fed7aa"),
                      "OK":("#16a34a","#f0fdf4","#bbf7d0")}
            pc,pbg,pborder=p_styles.get(priority,("#64748b","#f8fafc","#e2e8f0"))
            status_badge=(f"<span style='background:{pbg};color:{pc};"
                          f"border:1px solid {pborder};border-radius:4px;"
                          f"padding:3px 8px;font-size:10px;font-weight:700;"
                          f"white-space:nowrap;display:inline-block;'>{priority}</span>")
            row_bg={"HIGH":"#fffcfc","MEDIUM":"#fffdf9","OK":"#ffffff"}.get(priority,"#fff")
            stage_cells=""
            for stage in SUMMARY_STAGES:
                col=f'{stage}_secs'; limit=SLA_CONFIG[stage]['hard_limit']
                avg=row.get(col,0); is_breach=avg>limit and avg>0
                sc=sev_color(severity_label(avg,limit) if is_breach else "OK")
                cell_bg="#fff5f5" if is_breach else "#fff"
                wt="700" if is_breach else "400"
                stage_cells+=(f"<td style='padding:9px 10px;text-align:center;"
                               f"background:{cell_bg};border-bottom:1px solid #f1f5f9;"
                               f"border-left:1px solid #f0f4f8;'>"
                               f"<span style='font-size:12px;font-weight:{wt};color:{sc};'>"
                               f"{to_readable(avg)}</span></td>")
            lbl_r,col_r,bg_r=risk_meta(risk_val)
            risk_cell=(f"<td style='padding:7px 8px;text-align:center;background:{bg_r};"
                       f"border-bottom:1px solid #f1f5f9;border-left:1px solid #f0f4f8;'>"
                       f"{risk_badge_html(risk_val)}</td>")
            html+=(f"<tr style='background:{row_bg};'>"
                   f"<td style='padding:9px 14px;font-size:11px;font-weight:600;"
                   f"color:#1e293b;border-bottom:1px solid #f1f5f9;"
                   f"border-left:3px solid {pc};white-space:nowrap;'>"
                   f"{region_name}"
                   f"<div style='font-size:9px;color:#94a3b8;font-weight:400;margin-top:2px;'>"
                   f"{len(r_raw):,} orders</div></td>"
                   f"{simple_metric_td(region_otr,'9px 10px')}"
                   f"{stage_cells}"
                   f"{simple_metric_td(region_mod,'9px 10px')}"
                   f"{risk_cell}"
                   f"<td style='padding:9px 10px;text-align:center;"
                   f"border-bottom:1px solid #f1f5f9;'>{status_badge}</td></tr>")

        # Tier average row
        avg_cells=""
        for stage in SUMMARY_STAGES:
            col=f'{stage}_secs'; limit=SLA_CONFIG[stage]['hard_limit']
            vv=t_regions[t_regions[col]>0][col]
            tavg=round(vv.mean(),4) if len(vv)>0 else 0.0
            is_b=tavg>limit and tavg>0
            sc=sev_color(severity_label(tavg,limit) if is_b else "OK")
            cb="#fef3f2" if is_b else "#f0fdf4"
            avg_cells+=(f"<td style='padding:9px 10px;text-align:center;background:{cb};"
                        f"border-bottom:2px solid #e2e8f0;border-left:1px solid #e2e8f0;'>"
                        f"<span style='font-size:12px;font-weight:700;color:{sc};'>"
                        f"{to_readable(tavg)}</span>"
                        f"<div style='font-size:9px;color:#94a3b8;margin-top:2px;'>avg</div>"
                        f"</td>")
        otr_c=metric_color(tier_otr)
        otr_bg="#fff5f5" if tier_otr<50 else ("#fff8f0" if tier_otr<70 else "#f0fdf4")
        mod_c=metric_color(tier_mod)
        mod_bg="#fff5f5" if tier_mod<50 else ("#fff8f0" if tier_mod<70 else "#f0fdf4")
        avg_rs=int(t_regions['risk_score'].mean()) if 'risk_score' in t_regions.columns else 0
        lbl_a,col_a,bg_a=risk_meta(avg_rs)

        html+=(f"<tr style='background:#f8fafc;'>"
               f"<td style='padding:9px 14px;font-size:11px;font-weight:700;color:#334155;"
               f"border-bottom:2px solid #e2e8f0;border-left:3px solid {TIER_BG[tier]};'>"
               f"∅ Tier Average<div style='font-size:9px;color:#94a3b8;font-weight:400;"
               f"margin-top:2px;'>across {len(t_regions)} regions</div></td>"
               f"<td style='padding:9px 10px;text-align:center;background:{otr_bg};"
               f"border-bottom:2px solid #e2e8f0;border-left:1px solid #e2e8f0;'>"
               f"<span style='font-size:12px;font-weight:700;color:{otr_c};'>{tier_otr}%</span>"
               f"<div style='font-size:9px;color:#94a3b8;margin-top:2px;'>avg</div></td>"
               f"{avg_cells}"
               f"<td style='padding:9px 10px;text-align:center;background:{mod_bg};"
               f"border-bottom:2px solid #e2e8f0;border-left:1px solid #e2e8f0;'>"
               f"<span style='font-size:12px;font-weight:700;color:{mod_c};'>{tier_mod}%</span>"
               f"<div style='font-size:9px;color:#94a3b8;margin-top:2px;'>avg</div></td>"
               f"<td style='padding:7px 8px;text-align:center;background:{bg_a};"
               f"border-bottom:2px solid #e2e8f0;border-left:1px solid #e2e8f0;'>"
               f"{risk_badge_html(avg_rs)}</td>"
               f"<td style='padding:9px 10px;text-align:center;"
               f"border-bottom:2px solid #e2e8f0;font-size:10px;color:#94a3b8;'>—</td>"
               f"</tr>")

    html+="</tbody></table></div>"
    return html


# ═══════════════════════════════════════════════════════════════
#  STORE BREAKDOWN  (+ Risk + Anomaly flag)
# ═══════════════════════════════════════════════════════════════
def store_breakdown_html(region_stores_agg, region_raw, all_delivered_df):
    STAGE_SHORT={'OP to INP':'OP→INP','INP to PK':'INP→PK','PK to BIN':'PK→BIN',
                 'InhouseSLA':'InhouseSLA','BIN to RTS':'BIN→RTS'}
    if region_stores_agg.empty:
        return "<p style='color:#94a3b8;font-size:12px;padding:8px;'>No store data.</p>"

    store_rows=[(sr,get_priority(sr)) for _,sr in region_stores_agg.iterrows()
                if get_priority(sr)!='OK']
    if not store_rows:
        return """<div style='padding:10px 12px;font-size:11px;color:#16a34a;
                              background:#f0fdf4;border-radius:6px;
                              border:1px solid #bbf7d0;'>✓ All stores within SLA</div>"""
    store_rows.sort(key=lambda x:(priority_sort_key(x[1]),-x[0].get('InhouseSLA_secs',0)))

    stage_th="".join(f"""
        <th style='padding:8px 8px;text-align:center;min-width:115px;font-size:10px;'>
          <div style='font-weight:700;'>{STAGE_SHORT[s]}</div>
          <div style='font-size:9px;color:#94a3b8;font-weight:400;'>
            Limit:{to_readable(SLA_CONFIG[s]['hard_limit'])}
          </div>
        </th>""" for s in BREAKDOWN_STAGES)

    html=f"""
    <div style='overflow-x:auto;margin-top:4px;'>
    <table style='width:100%;border-collapse:collapse;font-size:11px;background:#fff;
                  border-radius:8px;overflow:hidden;border:1px solid #e2e8f0;'>
      <thead>
        <tr style='background:#f8fafc;'>
          <th style='padding:8px 12px;text-align:left;color:#475569;font-weight:700;
                     border-bottom:2px solid #e2e8f0;min-width:200px;font-size:10px;'>
            Store (HIGH → MEDIUM · InhouseSLA ↓)
          </th>
          {stage_th}
          {metric_th("OTR %","OTR=10","58px")}
          {metric_th("MOD %","MOD=1","58px")}
          <th style='padding:8px 8px;text-align:center;border-bottom:2px solid #e2e8f0;
                     min-width:78px;font-size:10px;'>Risk 🤖</th>
          <th style='padding:8px 8px;text-align:center;border-bottom:2px solid #e2e8f0;
                     min-width:68px;font-size:10px;'>Status</th>
        </tr>
      </thead>
      <tbody>"""

    for sr,priority in store_rows:
        store_name =sr['Store']
        s_raw      =region_raw[region_raw['Store']==store_name]
        inhouse_val=sr.get('InhouseSLA_secs',0)
        store_otr  =compute_otr_pct(all_delivered_df,store=store_name)
        store_mod  =compute_mod_pct(all_delivered_df,store=store_name)
        risk_val   =int(sr.get('risk_score',0))
        is_anom    =bool(sr.get('is_anomaly',False))
        anom_flag  =anomaly_flag_html(is_anom)
        lbl_s,col_s,bg_s=risk_meta(risk_val)
        p_styles={"HIGH":("#dc2626","#fef2f2","#fecaca"),
                  "MEDIUM":("#ea580c","#fff7ed","#fed7aa")}
        pc,pbg,pborder=p_styles.get(priority,("#64748b","#f8fafc","#e2e8f0"))
        status_badge=(f"<span style='background:{pbg};color:{pc};"
                      f"border:1px solid {pborder};border-radius:4px;"
                      f"padding:2px 7px;font-size:9px;font-weight:700;"
                      f"white-space:nowrap;'>{priority}</span>")
        row_bg="#fffcfc" if priority=="HIGH" else "#fffdf9"
        stage_cells="".join(
            build_stage_cell_store(stage,sr.get(f'{stage}_secs',0),s_raw,bar_width=100)
            for stage in BREAKDOWN_STAGES)

        html+=(f"<tr style='background:{row_bg};'>"
               f"<td style='padding:8px 12px;font-size:10px;font-weight:600;color:#1e293b;"
               f"border-bottom:1px solid #f1f5f9;white-space:nowrap;"
               f"border-left:3px solid {pc};'>"
               f"{store_name}{anom_flag}"
               f"<div style='font-size:9px;color:#94a3b8;font-weight:400;margin-top:1px;'>"
               f"{len(s_raw):,} orders &nbsp;·&nbsp; IH-SLA:{to_readable(inhouse_val)}"
               f"</div></td>"
               f"{stage_cells}"
               f"{simple_metric_td(store_otr,'8px 8px')}"
               f"{simple_metric_td(store_mod,'8px 8px')}"
               f"<td style='padding:7px 8px;text-align:center;background:{bg_s};"
               f"border-bottom:1px solid #f1f5f9;border-left:1px solid #f0f4f8;'>"
               f"{risk_badge_html(risk_val)}</td>"
               f"<td style='padding:8px 8px;text-align:center;"
               f"border-bottom:1px solid #f1f5f9;'>{status_badge}</td></tr>")

    html+="</tbody></table></div>"
    return html


# ═══════════════════════════════════════════════════════════════
#  REGIONWISE BREAKDOWN  (+ Risk column)
# ═══════════════════════════════════════════════════════════════
def regionwise_breakdown_html(region_df, store_df, raw_df, all_delivered_df):
    TIER_LABELS={'T1':'Tier 1 — Metro','T2':'Tier 2 — Major Cities','T3':'Tier 3 — Rural'}
    TIER_BG={'T1':'#1e293b','T2':'#1e3a5f','T3':'#1e4a3f'}
    STAGE_SHORT={'OP to INP':'OP→INP','INP to PK':'INP→PK','PK to BIN':'PK→BIN',
                 'InhouseSLA':'InhouseSLA','BIN to RTS':'BIN→RTS'}
    stage_th="".join(f"""
        <th style='padding:9px 8px;text-align:center;min-width:125px;'>
          <div style='font-weight:700;font-size:11px;'>{STAGE_SHORT[s]}</div>
          <div style='font-size:9px;color:#94a3b8;font-weight:400;margin-top:2px;'>
            Limit:{to_readable(SLA_CONFIG[s]['hard_limit'])}
          </div>
        </th>""" for s in BREAKDOWN_STAGES)

    html=f"""
    <div style='overflow-x:auto;'>
    <table style='width:100%;border-collapse:collapse;font-size:11px;background:#fff;
                  border-radius:10px;overflow:hidden;border:1px solid #e2e8f0;'>
      <thead>
        <tr style='background:#f1f5f9;'>
          <th style='padding:9px 14px;text-align:left;color:#475569;font-weight:700;
                     border-bottom:2px solid #e2e8f0;min-width:190px;'>
            Region
            <div style='font-size:9px;color:#94a3b8;font-weight:400;margin-top:2px;'>
              HIGH → MEDIUM · InhouseSLA ↓ · stores split
            </div>
          </th>
          {metric_th("OTR %","OTR=10 · &gt;50%/≤50%","90px")}
          {stage_th}
          {metric_th("MOD %","MOD=1","65px")}
          {metric_th("Risk 🤖","0-100","70px")}
          <th style='padding:9px 8px;text-align:center;border-bottom:2px solid #e2e8f0;
                     min-width:78px;'>Status</th>
        </tr>
      </thead>
      <tbody>"""

    uid_counter=[0]; total_shown=0

    for tier in ['T1','T2','T3']:
        t_regions=region_df[region_df['Tier']==tier].copy()
        if t_regions.empty: continue
        t_regions['_priority']=t_regions.apply(get_priority,axis=1)
        t_breach=t_regions[t_regions['_priority']!='OK'].copy()
        if t_breach.empty: continue
        t_breach['_psort']=t_breach['_priority'].map(priority_sort_key)
        t_breach=t_breach.sort_values(['_psort','InhouseSLA_secs'],
                                       ascending=[True,False]).reset_index(drop=True)
        n_high=int((t_breach['_priority']=='HIGH').sum())
        n_med =int((t_breach['_priority']=='MEDIUM').sum())

        html+=f"""
      <tr>
        <td colspan='{TOTAL_COLS}'
            style='background:{TIER_BG[tier]};color:#fff;font-size:11px;
                   font-weight:700;padding:7px 14px;letter-spacing:0.3px;'>
          {TIER_LABELS[tier]} &nbsp;·&nbsp; {len(t_breach)} breaching
          &nbsp;&nbsp;
          <span style='color:#fca5a5;font-size:10px;'>■ {n_high} HIGH</span>
          &nbsp;<span style='color:#fdba74;font-size:10px;'>■ {n_med} MEDIUM</span>
          &nbsp;&nbsp;
          <span style='color:#86efac;font-size:10px;font-weight:400;'>
            (sorted by InhouseSLA ↓)
          </span>
        </td>
      </tr>"""

        for _,region_row_data in t_breach.iterrows():
            uid_counter[0]+=1; uid=uid_counter[0]
            region_name=region_row_data['Region']
            priority   =region_row_data['_priority']
            inhouse_val=region_row_data.get('InhouseSLA_secs',0)
            r_raw      =raw_df[raw_df['Region']==region_name]
            r_stores   =store_df[store_df['Region']==region_name]
            region_otr =compute_otr_pct(all_delivered_df,region=region_name)
            region_mod =compute_mod_pct(all_delivered_df,region=region_name)
            ts_otr,ab_otr,be_otr=compute_otr_store_brackets(all_delivered_df,region=region_name)
            risk_val   =int(region_row_data.get('risk_score',0))
            lbl_r,col_r,bg_r=risk_meta(risk_val)
            total_shown+=1

            p_styles={"HIGH":("#dc2626","#fef2f2","#fecaca"),
                      "MEDIUM":("#ea580c","#fff7ed","#fed7aa")}
            pc,pbg,pborder=p_styles.get(priority,("#64748b","#f8fafc","#e2e8f0"))
            status_badge=(f"<span style='background:{pbg};color:{pc};"
                          f"border:1px solid {pborder};border-radius:4px;"
                          f"padding:3px 8px;font-size:10px;font-weight:700;"
                          f"white-space:nowrap;display:inline-block;'>{priority}</span>")
            row_bg="#fffcfc" if priority=="HIGH" else "#fffdf9"
            stage_cells="".join(
                build_stage_cell_region(stage,region_row_data.get(f'{stage}_secs',0),
                                        r_stores,bar_width=110)
                for stage in BREAKDOWN_STAGES)
            r_stores_breach=sum(1 for _,sr in r_stores.iterrows() if get_priority(sr)!='OK')
            store_html=store_breakdown_html(r_stores,r_raw,all_delivered_df)
            otr_cell=otr_region_td(region_otr,ab_otr,be_otr,ts_otr,
                                    pad="8px 8px",bar_width=85)
            mod_cell=simple_metric_td(region_mod,"8px 8px")
            risk_cell=(f"<td style='padding:7px 8px;text-align:center;background:{bg_r};"
                       f"border-bottom:1px solid #f1f5f9;border-left:1px solid #f0f4f8;'>"
                       f"{risk_badge_html(risk_val)}</td>")

            html+=f"""
      <tr style='background:{row_bg};' id='rrow-{uid}'>
        <td style='padding:0;border-bottom:1px solid #f1f5f9;border-left:3px solid {pc};'>
          <details id='rdet-{uid}' style='margin:0;'>
            <summary style='list-style:none;cursor:pointer;padding:9px 14px;
                            display:flex;align-items:flex-start;gap:8px;
                            user-select:none;flex-wrap:wrap;'>
              <span id='rarr-{uid}'
                    style='font-size:10px;color:#94a3b8;margin-top:2px;min-width:12px;'>▶
              </span>
              <div>
                <div style='font-size:12px;font-weight:700;color:#1e293b;'>
                  {region_name}
                </div>
                <div style='font-size:9px;color:#94a3b8;margin-top:2px;'>
                  {len(r_raw):,} orders &nbsp;·&nbsp;
                  {len(r_stores)} stores ({r_stores_breach} breaching)
                  &nbsp;·&nbsp; IH-SLA:{to_readable(inhouse_val)}
                </div>
              </div>
            </summary>
            <div style='padding:10px 14px 14px;background:#f8fafc;
                        border-top:1px dashed #e2e8f0;'>
              <div style='font-size:10px;font-weight:700;color:#475569;margin-bottom:8px;'>
                📋 Store Breakdown &nbsp;·&nbsp; Breaching stores only &nbsp;·&nbsp;
                HIGH → MEDIUM · InhouseSLA ↓
              </div>
              {store_html}
            </div>
          </details>
        </td>
        {otr_cell}
        {stage_cells}
        {mod_cell}
        {risk_cell}
        <td style='padding:8px 8px;text-align:center;border-bottom:1px solid #f1f5f9;'>
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

    if total_shown==0:
        html+=(f"<tr><td colspan='{TOTAL_COLS}' style='padding:24px;text-align:center;"
               f"color:#16a34a;font-size:13px;font-weight:600;'>"
               f"✓ All regions within SLA — No breaches found</td></tr>")
    html+="</tbody></table></div>"
    return html


# ═══════════════════════════════════════════════════════════════
#  BUILD FULL REPORT
# ═══════════════════════════════════════════════════════════════
def build_report(region_df, store_df, raw_df, all_delivered_df, filename, rca_map):
    now    =datetime.now().strftime("%d %b %Y, %I:%M %p")
    total_r=len(region_df); total_s=len(store_df); total_o=len(raw_df)

    html=f"""<!DOCTYPE html>
<html>
<head>
<style>
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{font-family:'Segoe UI',system-ui,Arial,sans-serif;
        background:#f8fafc;color:#1e293b;}}
  .wrap{{max-width:1260px;margin:auto;padding:16px;}}
  .hdr{{background:linear-gradient(135deg,#1e293b 0%,#334155 100%);
        color:#fff;border-radius:12px;padding:24px 28px;margin-bottom:20px;}}
  .hdr h1{{font-size:20px;font-weight:700;color:#fff;letter-spacing:0.3px;}}
  .hdr p{{font-size:11px;opacity:0.65;margin-top:4px;}}
  .sec-hdr{{display:flex;align-items:center;gap:10px;padding:10px 0;
            margin:20px 0 14px;border-bottom:2px solid #e2e8f0;}}
  .sec-hdr .title{{font-size:13px;font-weight:700;color:#1e293b;}}
  .sec-hdr .sub{{font-size:11px;color:#94a3b8;}}
  .info{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
         padding:10px 14px;font-size:11px;color:#475569;
         margin-bottom:16px;line-height:1.9;}}
  .legend{{display:inline-flex;align-items:center;gap:8px;font-size:10px;color:#64748b;}}
  .lg{{width:12px;height:12px;border-radius:2px;display:inline-block;}}
  .ts{{text-align:right;font-size:10px;color:#94a3b8;margin-bottom:12px;}}
  details summary::-webkit-details-marker{{display:none;}}
</style>
</head>
<body>
<div class='wrap'>
<div class='ts'>&#128337; {now} &nbsp;·&nbsp; {filename}</div>
<div class='hdr'>
  <h1>&#128202; SLA Breach — RCA Report + AI</h1>
  <p>Delivered · SLA Key=Yes ·
     {total_o:,} orders · {total_r} regions · {total_s} stores</p>
</div>

<div class='sec-hdr'>
  <span class='title'>&#128200; Performance Overview</span>
  <span class='sub'>Tier KPIs · Store counts · OTR% &amp; MOD% · AI Critical count</span>
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
  <b>AI Risk Score:</b>&nbsp;
  Isolation Forest anomaly detection + weighted breach severity (InhouseSLA 35 · BIN→RTS 25 · INP→PK 20 · OP→INP 10 · PK→BIN 10)
  &nbsp;·&nbsp; Multi-breach +20% penalty &nbsp;·&nbsp; Anomaly +15 bonus
  <br>
  <b>Risk bands:</b>&nbsp;
  <span style='color:#dc2626;font-weight:700;'>■ CRITICAL ≥70</span> &nbsp;|&nbsp;
  <span style='color:#ea580c;font-weight:700;'>■ HIGH 45-69</span> &nbsp;|&nbsp;
  <span style='color:#d97706;font-weight:700;'>■ MEDIUM 20-44</span> &nbsp;|&nbsp;
  <span style='color:#16a34a;font-weight:700;'>■ LOW &lt;20</span>
</div>

<div class='sec-hdr'>
  <span class='title'>🤖 AI Insights</span>
  <span class='sub'>Anomaly Detection · Risk Scoring · RCA Narratives</span>
</div>
{ai_section_html(store_df, all_delivered_df, rca_map)}

<div class='sec-hdr'>
  <span class='title'>&#128203; Summary Report</span>
  <span class='sub'>All regions · OTR% · Stages · MOD% · Risk 🤖 · Status</span>
</div>
{summary_report_html(region_df, raw_df, all_delivered_df)}

<div class='sec-hdr'>
  <span class='title'>&#128205; Regionwise Breakdown</span>
  <span class='sub'>
    Breaching only · OTR% store split + bar · MOD% · Risk 🤖 · Click → store drill-down
  </span>
  <div class='legend' style='margin-left:auto;'>
    <span class='lg' style='background:#16a34a;'></span>OK &nbsp;&nbsp;
    <span class='lg' style='background:#dc2626;'></span>Breach
  </div>
</div>
{regionwise_breakdown_html(region_df, store_df, raw_df, all_delivered_df)}
<br><br>
</div>
</body>
</html>"""
    return html


# ═══════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ═══════════════════════════════════════════════════════════════

# ── Sidebar — AI settings ──────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='background:linear-gradient(135deg,#4f46e5,#7c3aed);
                border-radius:8px;padding:12px 14px;color:#fff;margin-bottom:16px;'>
      <div style='font-size:14px;font-weight:700;'>🤖 AI Settings</div>
      <div style='font-size:10px;opacity:0.8;margin-top:2px;'>
        Anomaly detection + LLM RCA
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Anomaly Detection**")
    st.info(
        f"{'✅ scikit-learn ready' if SKLEARN_OK else '⚠️ Install scikit-learn for anomaly detection'}"
    )

    st.markdown("---")
    st.markdown("**LLM Provider** (optional)")
    llm_provider = st.selectbox(
        "Provider",
        ["None (Rule-Based RCA)", "OpenAI", "Gemini"],
        index=0, label_visibility="collapsed"
    )

    api_key = ""
    llm_model = ""
    if llm_provider == "OpenAI":
        api_key   = st.text_input("OpenAI API Key", type="password",
                                   placeholder="sk-...")
        llm_model = st.selectbox("Model",
                                  ["gpt-4o-mini","gpt-4o","gpt-3.5-turbo"],
                                  index=0)
        if not OPENAI_OK:
            st.warning("Run: `pip install openai`")
    elif llm_provider == "Gemini":
        api_key   = st.text_input("Gemini API Key", type="password",
                                   placeholder="AIza...")
        llm_model = "gemini-1.5-flash"
        if not GEMINI_OK:
            st.warning("Run: `pip install google-generativeai`")

    st.markdown("---")
    st.markdown("**Regions to Analyse**")
    n_regions = st.slider(
        "Top N breaching regions for RCA",
        min_value=1, max_value=20, value=5,
        help="Regions are selected by risk score (highest first)"
    )

    run_llm = False
    if llm_provider != "None (Rule-Based RCA)" and api_key:
        run_llm = st.button("🚀 Generate AI RCA", use_container_width=True)
    elif llm_provider != "None (Rule-Based RCA)":
        st.caption("Enter API key to enable LLM RCA")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:10px;color:#94a3b8;line-height:1.8;'>
    <b>AI Pipeline:</b><br>
    1️⃣ Isolation Forest → anomaly flags<br>
    2️⃣ Weighted scoring → risk 0-100<br>
    3️⃣ Rule-based RCA (always on)<br>
    4️⃣ LLM RCA → GPT/Gemini (optional)<br>
    5️⃣ Trend & urgency prediction
    </div>
    """, unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────
for k in ['rca_map','last_file','proc_done']:
    if k not in st.session_state:
        st.session_state[k] = {} if k=='rca_map' else None

# ── Main header ────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#1e293b,#334155);
            border-radius:12px;padding:22px 28px;color:#fff;
            font-family:Segoe UI,Arial,sans-serif;margin-bottom:24px;'>
  <h2 style='margin:0 0 6px;font-size:22px;letter-spacing:0.5px;color:#fff;'>
    📊 SLA Breach — RCA Report + AI
  </h2>
  <p style='margin:0;opacity:0.75;font-size:13px;'>
    Upload your Order Dump → Anomaly Detection · Risk Scoring ·
    LLM-powered RCA · Trend Prediction
  </p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📁 Upload Order Dump (Excel)",
    type=['xlsx','xls'],
    help="Sheet1 must contain: sa_name · OTR Status · SLA Key · OTR · MOD · stage cols"
)

if uploaded_file is not None:
    # Reset RCA cache when new file is uploaded
    if st.session_state['last_file'] != uploaded_file.name:
        st.session_state['rca_map']   = {}
        st.session_state['last_file'] = uploaded_file.name
        st.session_state['proc_done'] = False

    # ── Process data ───────────────────────────────────────────
    if not st.session_state['proc_done']:
        with st.spinner("⏳ Processing data…"):
            try:
                file_bytes = uploaded_file.read()

                raw_df, all_delivered_df = load_dump(file_bytes)

                store_df  = aggregate_by_store(raw_df)
                store_df  = flag_breaches(store_df)
                store_df  = compute_anomaly_scores(store_df)   # ← AI step 1
                store_df  = add_risk_scores(store_df)          # ← AI step 2

                region_df = aggregate_by_region(raw_df)
                region_df = flag_breaches(region_df)
                region_df = add_risk_scores(region_df)         # ← AI step 2 (region)

                st.session_state['raw_df']          = raw_df
                st.session_state['all_delivered_df']= all_delivered_df
                st.session_state['store_df']        = store_df
                st.session_state['region_df']       = region_df
                st.session_state['file_bytes']      = file_bytes
                st.session_state['proc_done']       = True
            except Exception as e:
                st.error(f"❌ Processing error: {e}")
                st.exception(e)
                st.stop()

    # ── Pull from session state ────────────────────────────────
    raw_df          = st.session_state['raw_df']
    all_delivered_df= st.session_state['all_delivered_df']
    store_df        = st.session_state['store_df']
    region_df       = st.session_state['region_df']

    n_anomalies = int(store_df['is_anomaly'].sum()) if 'is_anomaly' in store_df.columns else 0
    n_critical  = int((store_df['risk_score'] >= 70).sum())

    st.success(
        f"✅ Ready — {len(raw_df):,} orders · "
        f"{len(all_delivered_df):,} delivered · "
        f"{raw_df['Store'].nunique()} stores · "
        f"{raw_df['Region'].nunique()} regions · "
        f"🤖 {n_anomalies} anomalies · 🔴 {n_critical} critical"
    )

    # ── Rule-based RCA (always runs for top-N breach regions) ──
    breach_regions = (
        region_df[region_df.apply(get_priority, axis=1) != 'OK']
        .copy()
    )
    if 'risk_score' in breach_regions.columns:
        breach_regions = breach_regions.sort_values('risk_score', ascending=False)
    breach_regions = breach_regions.head(n_regions)

    TIER_LABEL_MAP = {'T1':'Tier 1 — Metro','T2':'Tier 2 — Major Cities',
                      'T3':'Tier 3 — Rural','Unknown':'Unknown'}

    rca_map = st.session_state['rca_map'].copy()

    # Always ensure rule-based RCA exists for top-N regions
    for _, rrow in breach_regions.iterrows():
        rname = rrow['Region']
        if rname not in rca_map:
            tier_lbl = TIER_LABEL_MAP.get(rrow.get('Tier',''), rrow.get('Tier',''))
            r_otr    = compute_otr_pct(all_delivered_df, region=rname)
            r_mod    = compute_mod_pct(all_delivered_df, region=rname)
            rs       = int(rrow.get('risk_score', 0))
            rca_map[rname] = generate_rule_based_rca(
                rname, rrow, tier_lbl, r_otr, r_mod, rs)
    st.session_state['rca_map'] = rca_map

    # ── LLM RCA (triggered by sidebar button) ─────────────────
    if run_llm and api_key and llm_provider != "None (Rule-Based RCA)":
        provider = llm_provider
        prog = st.progress(0, text="Generating AI RCA…")
        total_calls = len(breach_regions)

        for i, (_, rrow) in enumerate(breach_regions.iterrows()):
            rname    = rrow['Region']
            tier_lbl = TIER_LABEL_MAP.get(rrow.get('Tier',''), rrow.get('Tier',''))
            r_otr    = compute_otr_pct(all_delivered_df, region=rname)
            r_mod    = compute_mod_pct(all_delivered_df, region=rname)
            rs       = int(rrow.get('risk_score', 0))

            prog.progress((i+1)/total_calls,
                          text=f"LLM RCA: {rname} ({i+1}/{total_calls})…")

            # Build stores_data for prompt context
            r_stores_agg = store_df[store_df['Region']==rname]
            stores_data  = []
            for _,sr in r_stores_agg.nlargest(5,'risk_score').iterrows():
                stores_data.append({
                    'name':   sr['Store'],
                    'risk':   int(sr.get('risk_score',0)),
                    'ih':     to_readable(sr.get('InhouseSLA_secs',0)),
                    'br':     to_readable(sr.get('BIN to RTS_secs',0)),
                    'orders': sr.get('Total_Orders',0),
                    'anomaly':bool(sr.get('is_anomaly',False)),
                })

            rrow_dict  = rrow.to_dict()
            rrow_dict['_priority'] = get_priority(rrow)
            result     = generate_llm_rca(
                rname, rrow_dict, tier_lbl, r_otr, r_mod, rs,
                stores_data, provider, api_key, llm_model)
            result['risk_score'] = rs
            rca_map[rname]       = result

        prog.empty()
        st.session_state['rca_map'] = rca_map
        st.success(f"✅ LLM RCA generated for {total_calls} regions")

    # ── Build and render report ────────────────────────────────
    with st.spinner("Building HTML report…"):
        html_report = build_report(
            region_df, store_df, raw_df,
            all_delivered_df, uploaded_file.name,
            st.session_state['rca_map']
        )

    col1, col2 = st.columns([2,1])
    with col1:
        st.download_button(
            label="⬇️ Download Full Report (HTML)",
            data=html_report.encode('utf-8'),
            file_name=f"SLA_RCA_AI_{datetime.now().strftime('%d%b%Y_%H%M')}.html",
            mime="text/html",
            use_container_width=True
        )
    with col2:
        if st.button("🔄 Clear LLM Cache", use_container_width=True):
            st.session_state['rca_map'] = {}
            st.rerun()

    # ── Data summary expander ──────────────────────────────────
    with st.expander("📋 Data Summary + AI Risk Distribution", expanded=False):
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("All Delivered",     f"{len(all_delivered_df):,}")
        c2.metric("SLA Key=Yes",       f"{len(raw_df):,}")
        c3.metric("Stores",            f"{store_df['Store'].nunique():,}")
        c4.metric("Regions",           f"{region_df['Region'].nunique()}")
        c5.metric("🔴 Critical Stores",f"{n_critical}")
        c6.metric("⚠️ Anomalies",      f"{n_anomalies}")

        st.markdown("**Risk Score Distribution**")
        if 'risk_score' in store_df.columns:
            dist = {
                "CRITICAL (≥70)": int((store_df['risk_score']>=70).sum()),
                "HIGH (45-69)":   int(((store_df['risk_score']>=45)&(store_df['risk_score']<70)).sum()),
                "MEDIUM (20-44)": int(((store_df['risk_score']>=20)&(store_df['risk_score']<45)).sum()),
                "LOW (<20)":      int((store_df['risk_score']<20).sum()),
            }
            dist_df = pd.DataFrame(list(dist.items()), columns=["Band","Stores"])
            st.dataframe(dist_df, use_container_width=True, hide_index=True)

        st.markdown("**Top 10 Highest Risk Stores**")
        top10 = store_df.nlargest(10,'risk_score')[
            ['Store','Region','Tier','risk_score','anomaly_score',
             'is_anomaly','InhouseSLA_secs','BIN to RTS_secs','Total_Orders']
        ].copy()
        top10['InhouseSLA'] = top10['InhouseSLA_secs'].apply(to_readable)
        top10['BIN→RTS']    = top10['BIN to RTS_secs'].apply(to_readable)
        st.dataframe(
            top10[['Store','Region','Tier','risk_score','anomaly_score',
                   'is_anomaly','InhouseSLA','BIN→RTS','Total_Orders']],
            use_container_width=True, hide_index=True
        )

    st.markdown("---")
    components.html(html_report, height=9500, scrolling=True)

else:
    st.info("👆 Upload your Excel dump file to generate the AI-enhanced report")
    st.markdown("""
    <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
                padding:16px 20px;font-size:12px;color:#475569;line-height:2.1;'>
      <b>🤖 AI Features:</b><br>
      &nbsp;&nbsp;🔬 <b>Isolation Forest</b> — detects anomalous stores (unusual stage-time patterns)<br>
      &nbsp;&nbsp;📊 <b>Risk Score 0-100</b> — weighted breach severity per store &amp; region<br>
      &nbsp;&nbsp;📐 <b>Rule-Based RCA</b> — bottleneck identification + recommendations (no API)<br>
      &nbsp;&nbsp;🤖 <b>LLM RCA</b> — GPT-4o-mini or Gemini 1.5 Flash (add key in sidebar)<br>
      &nbsp;&nbsp;📈 <b>Trend Prediction</b> — IMPROVING / STABLE / WORSENING per region<br>
      &nbsp;&nbsp;⏱ <b>Urgency</b> — IMMEDIATE / 24H / 48H / MONITOR classification<br>
      <br>
      <b>Required columns:</b>&nbsp;
      <code>sa_name</code> · <code>OTR Status</code> · <code>SLA Key</code> ·
      <code>OTR</code> · <code>MOD</code> ·
      <code>Op to Inp 1</code> · <code>Inp to Pk 1</code> ·
      <code>Pk to Bin 1</code> · <code>Bin to Rts 1</code>
    </div>
    """, unsafe_allow_html=True)
