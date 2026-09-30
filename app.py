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

# ───────────────────────────────────────────────────────────────────
#  CONFIG
# ───────────────────────────────────────────────────────────────────
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

# ───────────────────────────────────────────────────────────────────
#  REGION / TIER MAPPING
# ───────────────────────────────────────────────────────────────────
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
    'VJY':'Vijayawada-Guntur','VIZ':'Visakhapatnam',
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

# ───────────────────────────────────────────────────────────────────
#  HELPERS
# ───────────────────────────────────────────────────────────────────
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
                return int(p[0])*3600 + int(p[1])*60 + int(float(p[2]))
            return 0
        if hasattr(val,'total_seconds'): return max(0,int(val.total_seconds()))
        if hasattr(val,'hour'): return val.hour*3600+val.minute*60+val.second
        return 0
    except: return 0

def to_readable(secs):
    if not secs or secs <= 0: return "0s"
    s = int(secs)
    if s >= 3600: return f"{s//3600}h {(s%3600)//60}m {s%60}s"
    elif s >= 60: return f"{s//60}m {s%60}s"
    return f"{s}s"

def severity_label(actual, hard_limit):
    if actual <= 0: return "OK"
    pct = ((actual - hard_limit) / hard_limit) * 100
    if pct >= 100: return "CRITICAL"
    elif pct >= 50: return "HIGH"
    return "BREACH"

def sev_color(sev):
    return {"CRITICAL":"#dc2626","HIGH":"#ea580c",
            "BREACH":"#d97706","OK":"#16a34a"}.get(sev,"#64748b")

def get_priority(row_data):
    ih = bool(row_data.get('InhouseSLA_breach', False))
    br = bool(row_data.get('BIN to RTS_breach', False))
    if ih and br:  return "HIGH"
    elif ih or br: return "MEDIUM"
    return "OK"

def priority_sort_key(p):
    return {"HIGH":0,"MEDIUM":1,"OK":2}.get(p, 3)

def get_stage_pct_orders(df, stage, store=None):
    col   = f'{stage}_secs'
    limit = SLA_CONFIG[stage]['hard_limit']
    ds    = df if store is None else df[df['Store']==store]
    if col not in ds.columns: return 0,0,0,0.0,0.0
    valid = ds[ds[col]>0][col]
    total = len(valid)
    if total == 0: return 0,0,0,0.0,0.0
    bc = int((valid>limit).sum())
    wc = total - bc
    return total, wc, bc, round(wc/total*100,1), round(bc/total*100,1)

def get_stage_pct_stores(stores_agg_df, stage):
    col   = f'{stage}_secs'
    limit = SLA_CONFIG[stage]['hard_limit']
    if col not in stores_agg_df.columns or stores_agg_df.empty:
        return 0,0,0,0.0,0.0
    valid = stores_agg_df[stores_agg_df[col]>0]
    total = len(valid)
    if total == 0: return 0,0,0,0.0,0.0
    bc = int((valid[col]>limit).sum())
    wc = total - bc
    return total, wc, bc, round(wc/total*100,1), round(bc/total*100,1)

def split_bar(wp, bp, wn=0, bn=0, width=120):
    if wp == 0 and bp == 0:
        return "<span style='color:#94a3b8;font-size:10px;'>—</span>"
    gp    = max(0.0, min(100.0, wp))
    rp    = max(0.0, min(100.0, bp))
    total = wn + bn

    def pct_lbl(pct):
        if pct >= 10: return f"{pct:.0f}%"
        if pct >= 5:  return f"{pct:.0f}%"
        return ""

    gmin      = "2px" if gp > 0 else "0"
    rmin      = "2px" if rp > 0 else "0"
    total_lbl = (f"<span style='font-size:9px;color:#94a3b8;"
                 f"margin-left:4px;vertical-align:middle;'>{total}</span>"
                 if total > 0 else "")

    return f"""<div style='display:inline-flex;width:{width}px;height:20px;
                border-radius:4px;overflow:hidden;vertical-align:middle;
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
    </div>{total_lbl}"""

# ───────────────────────────────────────────────────────────────────
#  LOAD & FILTER
# ───────────────────────────────────────────────────────────────────
def load_dump(fb):
    df = pd.read_excel(io.BytesIO(fb), sheet_name='Sheet1', header=0)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={'sa_name':'Store'})
    df['_otr']  = df['OTR Status'].astype(str).str.strip().str.lower()
    df['_slak'] = df['SLA Key'].astype(str).str.strip()
    df = df[(df['_otr']=='delivered') & (df['_slak']=='Yes')].copy()
    df.drop(columns=['_otr','_slak'], inplace=True)
    df = df.rename(columns=DUMP_COL_MAP)
    df = df[df['Store'].notna()]
    df = df[df['Store'].astype(str).str.strip().str.startswith('DS-')]
    df = df.reset_index(drop=True)
    for col in STAGE_COLS:
        df[f'{col}_secs'] = df[col].apply(time_to_seconds) if col in df.columns else 0
    df['InhouseSLA_secs'] = (df['OP to INP_secs'] +
                              df['INP to PK_secs'] +
                              df['PK to BIN_secs'])
    df['Region'] = df['Store'].apply(extract_region)
    df['Tier']   = df['Region'].apply(get_tier)
    return df

# ───────────────────────────────────────────────────────────────────
#  AGGREGATE
# ───────────────────────────────────────────────────────────────────
def aggregate_by_store(df):
    records = []
    for store, grp in df.groupby('Store'):
        row = {'Store':store,'Region':grp['Region'].iloc[0],
               'Tier':grp['Tier'].iloc[0],'Total_Orders':len(grp)}
        for col in INDIVIDUAL_SECS:
            nz = grp[grp[col]>0][col]
            row[col] = round(nz.mean(),4) if len(nz)>0 else 0.0
        row['InhouseSLA_secs'] = (
            row.get('OP to INP_secs',0) +
            row.get('INP to PK_secs',0) +
            row.get('PK to BIN_secs',0)
        )
        records.append(row)
    return pd.DataFrame(records)

def aggregate_by_region(df):
    records = []
    for region, grp in df.groupby('Region'):
        if region in ('Unknown','#N/A','','nan'): continue
        row = {'Region':region,'Tier':grp['Tier'].iloc[0],
               'Total_Orders':len(grp)}
        for col in INDIVIDUAL_SECS:
            nz = grp[grp[col]>0][col]
            row[col] = round(nz.mean(),4) if len(nz)>0 else 0.0
        row['InhouseSLA_secs'] = (
            row.get('OP to INP_secs',0) +
            row.get('INP to PK_secs',0) +
            row.get('PK to BIN_secs',0)
        )
        records.append(row)
    return pd.DataFrame(records)

def flag_breaches(df):
    for stage in list(SLA_CONFIG.keys()):
        col   = f'{stage}_secs'
        limit = SLA_CONFIG[stage]['hard_limit']
        if col in df.columns:
            df[f'{stage}_breach'] = (df[col]>limit) & (df[col]>0)
        else:
            df[f'{stage}_breach'] = False
    return df

# ───────────────────────────────────────────────────────────────────
#  STAGE CELL BUILDERS
# ───────────────────────────────────────────────────────────────────
def build_stage_cell_region(stage, avg, stores_agg_df, bar_width=110):
    limit     = SLA_CONFIG[stage]['hard_limit']
    is_breach = avg > limit and avg > 0
    sev       = severity_label(avg, limit) if is_breach else "OK"
    sc        = sev_color(sev)
    cell_bg   = "#fff5f5" if is_breach else "#fafafa"
    wt        = "700" if is_breach else "500"
    total_, wn, bn, wp, bp = get_stage_pct_stores(stores_agg_df, stage)
    if total_ > 0:
        counts_html = f"""
        <div style='display:flex;gap:4px;justify-content:center;margin-top:3px;flex-wrap:wrap;'>
          <span style='font-size:9px;color:#16a34a;font-weight:600;background:#f0fdf4;
                       padding:1px 5px;border-radius:3px;white-space:nowrap;'>OK:{wn}</span>
          <span style='font-size:9px;color:#dc2626;font-weight:600;background:#fef2f2;
                       padding:1px 5px;border-radius:3px;white-space:nowrap;'>B:{bn}</span>
        </div>
        <div style='margin-top:3px;text-align:center;'>
          {split_bar(wp, bp, wn, bn, width=bar_width)}
        </div>"""
    else:
        counts_html = "<div style='font-size:9px;color:#cbd5e1;margin-top:3px;'>—</div>"
    return f"""
    <td style='padding:8px 7px;text-align:center;background:{cell_bg};
               border-bottom:1px solid #f1f5f9;border-left:1px solid #f0f4f8;'>
      <div style='font-size:12px;font-weight:{wt};color:{sc};'>{to_readable(avg)}</div>
      {counts_html}
    </td>"""

def build_stage_cell_store(stage, avg, raw_df_subset, bar_width=100):
    limit     = SLA_CONFIG[stage]['hard_limit']
    is_breach = avg > limit and avg > 0
    sev       = severity_label(avg, limit) if is_breach else "OK"
    sc        = sev_color(sev)
    cell_bg   = "#fff5f5" if is_breach else "#fafafa"
    wt        = "700" if is_breach else "500"
    total_, wn, bn, wp, bp = get_stage_pct_orders(raw_df_subset, stage)
    if total_ > 0:
        counts_html = f"""
        <div style='display:flex;gap:4px;justify-content:center;margin-top:3px;flex-wrap:wrap;'>
          <span style='font-size:9px;color:#16a34a;font-weight:600;background:#f0fdf4;
                       padding:1px 5px;border-radius:3px;white-space:nowrap;'>OK:{wn}</span>
          <span style='font-size:9px;color:#dc2626;font-weight:600;background:#fef2f2;
                       padding:1px 5px;border-radius:3px;white-space:nowrap;'>B:{bn}</span>
        </div>
        <div style='margin-top:3px;text-align:center;'>
          {split_bar(wp, bp, wn, bn, width=bar_width)}
        </div>"""
    else:
        counts_html = "<div style='font-size:9px;color:#cbd5e1;margin-top:3px;'>—</div>"
    return f"""
    <td style='padding:8px 7px;text-align:center;background:{cell_bg};
               border-bottom:1px solid #f1f5f9;border-left:1px solid #f0f4f8;'>
      <div style='font-size:12px;font-weight:{wt};color:{sc};'>{to_readable(avg)}</div>
      {counts_html}
    </td>"""

# ───────────────────────────────────────────────────────────────────
#  SUMMARY REPORT
# ───────────────────────────────────────────────────────────────────
def summary_report_html(region_df, raw_df):
    TIER_LABELS = {'T1':'Tier 1 — Metro','T2':'Tier 2 — Major Cities','T3':'Tier 3 — Rural'}
    TIER_BG     = {'T1':'#1e293b','T2':'#1e3a5f','T3':'#1e4a3f'}
    STAGE_SHORT = {
        'OP to INP':'OP → INP','INP to PK':'INP → PK',
        'PK to BIN':'PK → BIN','InhouseSLA':'InhouseSLA','BIN to RTS':'BIN → RTS',
    }
    stage_th = "".join(f"""
        <th style='padding:9px 12px;text-align:center;min-width:110px;
                   color:#475569;font-weight:600;border-bottom:2px solid #e2e8f0;'>
          <div>{STAGE_SHORT[s]}</div>
          <div style='font-size:9px;color:#94a3b8;font-weight:400;margin-top:2px;'>
            Limit: {to_readable(SLA_CONFIG[s]['hard_limit'])}
          </div>
        </th>""" for s in SUMMARY_STAGES)

    html = f"""
    <div style='overflow-x:auto;'>
    <table style='width:100%;border-collapse:collapse;font-size:11px;
                  background:#fff;border-radius:10px;overflow:hidden;
                  border:1px solid #e2e8f0;'>
      <thead>
        <tr style='background:#f1f5f9;'>
          <th style='padding:9px 14px;text-align:left;color:#475569;font-weight:600;
                     border-bottom:2px solid #e2e8f0;min-width:180px;'>
            Region
            <div style='font-size:9px;color:#94a3b8;font-weight:400;margin-top:2px;'>
              HIGH → MEDIUM → OK &nbsp;·&nbsp; InhouseSLA ↓
            </div>
          </th>
          {stage_th}
          <th style='padding:9px 12px;text-align:center;border-bottom:2px solid #e2e8f0;
                     min-width:80px;color:#475569;font-weight:600;'>Status</th>
        </tr>
      </thead>
      <tbody>"""

    for tier in ['T1','T2','T3']:
        t_regions = region_df[region_df['Tier']==tier].copy()
        if t_regions.empty: continue
        t_regions['_priority'] = t_regions.apply(get_priority, axis=1)
        t_regions['_psort']    = t_regions['_priority'].map(priority_sort_key)
        t_regions = t_regions.sort_values(
            ['_psort','InhouseSLA_secs'], ascending=[True,False]
        ).reset_index(drop=True)
        n_high   = int((t_regions['_priority']=='HIGH').sum())
        n_medium = int((t_regions['_priority']=='MEDIUM').sum())
        n_ok     = int((t_regions['_priority']=='OK').sum())

        html += f"""
      <tr>
        <td colspan='{len(SUMMARY_STAGES)+2}'
            style='background:{TIER_BG[tier]};color:#fff;font-size:11px;
                   font-weight:700;padding:7px 14px;letter-spacing:0.3px;'>
          {TIER_LABELS[tier]} &nbsp;·&nbsp; {len(t_regions)} regions
          &nbsp;&nbsp;
          <span style='color:#fca5a5;font-size:10px;'>■ {n_high} HIGH</span>
          &nbsp;
          <span style='color:#fdba74;font-size:10px;'>■ {n_medium} MEDIUM</span>
          &nbsp;
          <span style='color:#86efac;font-size:10px;'>■ {n_ok} OK</span>
        </td>
      </tr>"""

        for _, row in t_regions.iterrows():
            region_name = row['Region']
            priority    = row['_priority']
            r_raw       = raw_df[raw_df['Region']==region_name]
            p_styles = {
                "HIGH":   ("#dc2626","#fef2f2","#fecaca"),
                "MEDIUM": ("#ea580c","#fff7ed","#fed7aa"),
                "OK":     ("#16a34a","#f0fdf4","#bbf7d0"),
            }
            pc, pbg, pborder = p_styles.get(priority, ("#64748b","#f8fafc","#e2e8f0"))
            status_badge = f"""
            <span style='background:{pbg};color:{pc};border:1px solid {pborder};
                         border-radius:4px;padding:3px 8px;font-size:10px;
                         font-weight:700;white-space:nowrap;display:inline-block;'>
              {priority}
            </span>"""
            row_bg = {"HIGH":"#fffcfc","MEDIUM":"#fffdf9","OK":"#ffffff"}.get(priority,"#ffffff")
            stage_cells = ""
            for stage in SUMMARY_STAGES:
                col       = f'{stage}_secs'
                limit     = SLA_CONFIG[stage]['hard_limit']
                avg       = row.get(col, 0)
                is_breach = avg > limit and avg > 0
                sev       = severity_label(avg, limit) if is_breach else "OK"
                sc        = sev_color(sev)
                cell_bg   = "#fff5f5" if is_breach else "#fff"
                wt        = "700" if is_breach else "400"
                stage_cells += f"""
                <td style='padding:9px 10px;text-align:center;background:{cell_bg};
                           border-bottom:1px solid #f1f5f9;border-left:1px solid #f0f4f8;'>
                  <span style='font-size:12px;font-weight:{wt};color:{sc};'>
                    {to_readable(avg)}
                  </span>
                </td>"""

            html += f"""
      <tr style='background:{row_bg};'>
        <td style='padding:9px 14px;font-size:11px;font-weight:600;color:#1e293b;
                   border-bottom:1px solid #f1f5f9;border-left:3px solid {pc};
                   white-space:nowrap;'>
          {region_name}
          <div style='font-size:9px;color:#94a3b8;font-weight:400;margin-top:2px;'>
            {len(r_raw):,} orders
          </div>
        </td>
        {stage_cells}
        <td style='padding:9px 10px;text-align:center;border-bottom:1px solid #f1f5f9;'>
          {status_badge}
        </td>
      </tr>"""

        # Tier Average Row
        avg_stage_cells = ""
        for stage in SUMMARY_STAGES:
            col        = f'{stage}_secs'
            limit      = SLA_CONFIG[stage]['hard_limit']
            valid_vals = t_regions[t_regions[col]>0][col]
            tier_avg   = round(valid_vals.mean(), 4) if len(valid_vals)>0 else 0.0
            is_breach  = tier_avg > limit and tier_avg > 0
            sev        = severity_label(tier_avg, limit) if is_breach else "OK"
            sc         = sev_color(sev)
            cell_bg    = "#fef3f2" if is_breach else "#f0fdf4"
            avg_stage_cells += f"""
            <td style='padding:9px 10px;text-align:center;background:{cell_bg};
                       border-bottom:2px solid #e2e8f0;border-left:1px solid #e2e8f0;'>
              <span style='font-size:12px;font-weight:700;color:{sc};'>
                {to_readable(tier_avg)}
              </span>
              <div style='font-size:9px;color:#94a3b8;margin-top:2px;'>avg</div>
            </td>"""

        html += f"""
      <tr style='background:#f8fafc;'>
        <td style='padding:9px 14px;font-size:11px;font-weight:700;color:#334155;
                   border-bottom:2px solid #e2e8f0;border-left:3px solid {TIER_BG[tier]};
                   white-space:nowrap;'>
          ∅ Tier Average
          <div style='font-size:9px;color:#94a3b8;font-weight:400;margin-top:2px;'>
            across {len(t_regions)} regions
          </div>
        </td>
        {avg_stage_cells}
        <td style='padding:9px 10px;text-align:center;border-bottom:2px solid #e2e8f0;
                   font-size:10px;color:#94a3b8;'>—</td>
      </tr>"""

    html += "</tbody></table></div>"
    return html

# ───────────────────────────────────────────────────────────────────
#  STORE BREAKDOWN TABLE
# ───────────────────────────────────────────────────────────────────
def store_breakdown_html(region_stores_agg, region_raw):
    STAGE_SHORT = {
        'OP to INP':'OP → INP','INP to PK':'INP → PK',
        'PK to BIN':'PK → BIN','InhouseSLA':'InhouseSLA','BIN to RTS':'BIN → RTS',
    }
    if region_stores_agg.empty:
        return "<p style='color:#94a3b8;font-size:12px;padding:8px;'>No store data.</p>"
    store_rows = [(sr, get_priority(sr)) for _, sr in region_stores_agg.iterrows()
                  if get_priority(sr) != 'OK']
    if not store_rows:
        return """<div style='padding:10px 12px;font-size:11px;color:#16a34a;
                              background:#f0fdf4;border-radius:6px;border:1px solid #bbf7d0;'>
                    ✓ All stores within SLA
                  </div>"""
    store_rows.sort(key=lambda x: (priority_sort_key(x[1]), -x[0].get('InhouseSLA_secs',0)))

    stage_th = "".join(f"""
        <th style='padding:8px 8px;text-align:center;min-width:120px;font-size:10px;'>
          <div style='font-weight:700;'>{STAGE_SHORT[s]}</div>
          <div style='font-size:9px;color:#94a3b8;font-weight:400;margin-top:1px;'>
            Limit:{to_readable(SLA_CONFIG[s]['hard_limit'])}
          </div>
        </th>""" for s in BREAKDOWN_STAGES)

    html = f"""
    <div style='overflow-x:auto;margin-top:4px;'>
    <table style='width:100%;border-collapse:collapse;font-size:11px;
                  background:#fff;border-radius:8px;overflow:hidden;border:1px solid #e2e8f0;'>
      <thead>
        <tr style='background:#f8fafc;'>
          <th style='padding:8px 12px;text-align:left;color:#475569;font-weight:700;
                     border-bottom:2px solid #e2e8f0;min-width:200px;font-size:10px;'>
            Store
            <span style='color:#94a3b8;font-weight:400;margin-left:4px;'>
              (HIGH → MEDIUM · InhouseSLA ↓ · orders split)
            </span>
          </th>
          {stage_th}
          <th style='padding:8px 8px;text-align:center;border-bottom:2px solid #e2e8f0;
                     min-width:70px;font-size:10px;'>Status</th>
        </tr>
      </thead>
      <tbody>"""

    for sr, priority in store_rows:
        store_name  = sr['Store']
        s_raw       = region_raw[region_raw['Store']==store_name]
        inhouse_val = sr.get('InhouseSLA_secs', 0)
        p_styles = {
            "HIGH":   ("#dc2626","#fef2f2","#fecaca"),
            "MEDIUM": ("#ea580c","#fff7ed","#fed7aa"),
        }
        pc, pbg, pborder = p_styles.get(priority, ("#64748b","#f8fafc","#e2e8f0"))
        status_badge = f"""
        <span style='background:{pbg};color:{pc};border:1px solid {pborder};
                     border-radius:4px;padding:2px 7px;font-size:9px;font-weight:700;
                     white-space:nowrap;'>{priority}</span>"""
        row_bg = "#fffcfc" if priority=="HIGH" else "#fffdf9"
        stage_cells = "".join(
            build_stage_cell_store(stage, sr.get(f'{stage}_secs',0), s_raw, bar_width=100)
            for stage in BREAKDOWN_STAGES
        )
        html += f"""
      <tr style='background:{row_bg};'>
        <td style='padding:8px 12px;font-size:10px;font-weight:600;color:#1e293b;
                   border-bottom:1px solid #f1f5f9;white-space:nowrap;
                   border-left:3px solid {pc};'>
          {store_name}
          <div style='font-size:9px;color:#94a3b8;font-weight:400;margin-top:1px;'>
            {len(s_raw):,} orders &nbsp;·&nbsp; InhouseSLA: {to_readable(inhouse_val)}
          </div>
        </td>
        {stage_cells}
        <td style='padding:8px 8px;text-align:center;border-bottom:1px solid #f1f5f9;'>
          {status_badge}
        </td>
      </tr>"""

    html += "</tbody></table></div>"
    return html

# ───────────────────────────────────────────────────────────────────
#  REGIONWISE BREAKDOWN
# ───────────────────────────────────────────────────────────────────
def regionwise_breakdown_html(region_df, store_df, raw_df):
    TIER_LABELS = {'T1':'Tier 1 — Metro','T2':'Tier 2 — Major Cities','T3':'Tier 3 — Rural'}
    TIER_BG     = {'T1':'#1e293b','T2':'#1e3a5f','T3':'#1e4a3f'}
    STAGE_SHORT = {
        'OP to INP':'OP → INP','INP to PK':'INP → PK',
        'PK to BIN':'PK → BIN','InhouseSLA':'InhouseSLA','BIN to RTS':'BIN → RTS',
    }
    stage_th = "".join(f"""
        <th style='padding:9px 8px;text-align:center;min-width:130px;'>
          <div style='font-weight:700;'>{STAGE_SHORT[s]}</div>
          <div style='font-size:9px;color:#94a3b8;font-weight:400;margin-top:2px;'>
            Limit: {to_readable(SLA_CONFIG[s]['hard_limit'])}
          </div>
        </th>""" for s in BREAKDOWN_STAGES)

    html = f"""
    <div style='overflow-x:auto;'>
    <table style='width:100%;border-collapse:collapse;font-size:11px;
                  background:#fff;border-radius:10px;overflow:hidden;border:1px solid #e2e8f0;'>
      <thead>
        <tr style='background:#f1f5f9;'>
          <th style='padding:9px 14px;text-align:left;color:#475569;font-weight:700;
                     border-bottom:2px solid #e2e8f0;min-width:190px;'>
            Region
            <div style='font-size:9px;color:#94a3b8;font-weight:400;margin-top:2px;'>
              HIGH → MEDIUM · InhouseSLA ↓ · stores split
            </div>
          </th>
          {stage_th}
          <th style='padding:9px 8px;text-align:center;border-bottom:2px solid #e2e8f0;
                     min-width:80px;'>Status</th>
        </tr>
      </thead>
      <tbody>"""

    uid_counter = [0]
    total_shown = 0

    for tier in ['T1','T2','T3']:
        t_regions = region_df[region_df['Tier']==tier].copy()
        if t_regions.empty: continue
        t_regions['_priority'] = t_regions.apply(get_priority, axis=1)
        t_breach = t_regions[t_regions['_priority']!='OK'].copy()
        if t_breach.empty: continue
        t_breach['_psort'] = t_breach['_priority'].map(priority_sort_key)
        t_breach = t_breach.sort_values(
            ['_psort','InhouseSLA_secs'], ascending=[True,False]
        ).reset_index(drop=True)
        n_high   = int((t_breach['_priority']=='HIGH').sum())
        n_medium = int((t_breach['_priority']=='MEDIUM').sum())

        html += f"""
      <tr>
        <td colspan='{len(BREAKDOWN_STAGES)+2}'
            style='background:{TIER_BG[tier]};color:#fff;font-size:11px;
                   font-weight:700;padding:7px 14px;letter-spacing:0.3px;'>
          {TIER_LABELS[tier]} &nbsp;·&nbsp; {len(t_breach)} breaching
          &nbsp;&nbsp;
          <span style='color:#fca5a5;font-size:10px;'>■ {n_high} HIGH</span>
          &nbsp;
          <span style='color:#fdba74;font-size:10px;'>■ {n_medium} MEDIUM</span>
          &nbsp;&nbsp;
          <span style='color:#86efac;font-size:10px;font-weight:400;'>
            (sorted by InhouseSLA ↓)
          </span>
        </td>
      </tr>"""

        for _, region_row_data in t_breach.iterrows():
            uid_counter[0] += 1
            uid          = uid_counter[0]
            region_name  = region_row_data['Region']
            priority     = region_row_data['_priority']
            inhouse_val  = region_row_data.get('InhouseSLA_secs', 0)
            r_raw        = raw_df[raw_df['Region']==region_name]
            r_stores     = store_df[store_df['Region']==region_name]
            total_shown += 1
            p_styles = {
                "HIGH":   ("#dc2626","#fef2f2","#fecaca"),
                "MEDIUM": ("#ea580c","#fff7ed","#fed7aa"),
            }
            pc, pbg, pborder = p_styles.get(priority, ("#64748b","#f8fafc","#e2e8f0"))
            status_badge = f"""
            <span style='background:{pbg};color:{pc};border:1px solid {pborder};
                         border-radius:4px;padding:3px 8px;font-size:10px;font-weight:700;
                         white-space:nowrap;display:inline-block;'>{priority}</span>"""
            row_bg = "#fffcfc" if priority=="HIGH" else "#fffdf9"
            stage_cells = "".join(
                build_stage_cell_region(stage, region_row_data.get(f'{stage}_secs',0),
                                        r_stores, bar_width=110)
                for stage in BREAKDOWN_STAGES
            )
            r_stores_breach = sum(1 for _, sr in r_stores.iterrows() if get_priority(sr)!='OK')
            store_html = store_breakdown_html(r_stores, r_raw)

            html += f"""
      <tr style='background:{row_bg};' id='rrow-{uid}'>
        <td style='padding:0;border-bottom:1px solid #f1f5f9;border-left:3px solid {pc};'>
          <details id='rdet-{uid}' style='margin:0;'>
            <summary style='list-style:none;cursor:pointer;padding:9px 14px;
                            display:flex;align-items:flex-start;gap:8px;
                            user-select:none;flex-wrap:wrap;'>
              <span id='rarr-{uid}'
                    style='font-size:10px;color:#94a3b8;margin-top:2px;min-width:12px;'>▶</span>
              <div>
                <div style='font-size:12px;font-weight:700;color:#1e293b;'>{region_name}</div>
                <div style='font-size:9px;color:#94a3b8;margin-top:2px;'>
                  {len(r_raw):,} orders &nbsp;·&nbsp;
                  {len(r_stores)} stores ({r_stores_breach} breaching)
                  &nbsp;·&nbsp; InhouseSLA: {to_readable(inhouse_val)}
                </div>
              </div>
            </summary>
            <div style='padding:10px 14px 14px;background:#f8fafc;border-top:1px dashed #e2e8f0;'>
              <div style='font-size:10px;font-weight:700;color:#475569;margin-bottom:8px;'>
                📋 Store Breakdown &nbsp;·&nbsp; Breaching stores only &nbsp;·&nbsp;
                HIGH → MEDIUM · InhouseSLA ↓
              </div>
              {store_html}
            </div>
          </details>
        </td>
        {stage_cells}
        <td style='padding:8px 8px;text-align:center;border-bottom:1px solid #f1f5f9;'>
          {status_badge}
        </td>
      </tr>
      <script>
      (function(){{
        var d = document.getElementById('rdet-{uid}');
        var a = document.getElementById('rarr-{uid}');
        if(d && a){{
          d.addEventListener('toggle', function(){{
            a.textContent = d.open ? '▼' : '▶';
          }});
        }}
      }})();
      </script>"""

    if total_shown == 0:
        html += f"""
      <tr>
        <td colspan='{len(BREAKDOWN_STAGES)+2}'
            style='padding:24px;text-align:center;color:#16a34a;
                   font-size:13px;font-weight:600;'>
          ✓ All regions are within SLA — No breaches found
        </td>
      </tr>"""

    html += "</tbody></table></div>"
    return html

# ───────────────────────────────────────────────────────────────────
#  BUILD FULL REPORT HTML STRING
# ───────────────────────────────────────────────────────────────────
def build_report(region_df, store_df, raw_df, filename):
    now     = datetime.now().strftime("%d %b %Y, %I:%M %p")
    total_r = len(region_df)
    total_s = len(store_df)
    total_o = len(raw_df)

    n_high=0; n_medium=0; n_ok=0; n_ih=0; n_br=0
    for _, row in region_df.iterrows():
        p = get_priority(row)
        if p=='HIGH':     n_high   += 1
        elif p=='MEDIUM': n_medium += 1
        else:             n_ok     += 1
        if row.get('InhouseSLA_breach', False): n_ih += 1
        if row.get('BIN to RTS_breach', False): n_br += 1

    html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
  * {{ box-sizing:border-box;margin:0;padding:0; }}
  body {{ font-family:'Segoe UI',system-ui,Arial,sans-serif;
          background:#f8fafc;color:#1e293b; }}
  .wrap {{ max-width:1200px;margin:auto;padding:16px; }}
  .hdr  {{ background:linear-gradient(135deg,#1e293b 0%,#334155 100%);
           color:#fff;border-radius:12px;padding:24px 28px;margin-bottom:20px; }}
  .hdr h1 {{ font-size:20px;font-weight:700;letter-spacing:0.3px;color:#fff; }}
  .hdr p  {{ font-size:11px;opacity:0.65;margin-top:4px; }}
  .kpis {{ display:flex;gap:10px;flex-wrap:wrap;margin-bottom:20px; }}
  .kpi  {{ flex:1;min-width:100px;background:#fff;border-radius:8px;
           border:1px solid #e2e8f0;padding:14px 16px;text-align:center; }}
  .kpi .v {{ font-size:26px;font-weight:700; }}
  .kpi .l {{ font-size:10px;color:#94a3b8;margin-top:3px; }}
  .sec-hdr {{ display:flex;align-items:center;gap:10px;padding:10px 0;
              margin:20px 0 14px;border-bottom:2px solid #e2e8f0; }}
  .sec-hdr .title {{ font-size:13px;font-weight:700;color:#1e293b; }}
  .sec-hdr .sub   {{ font-size:11px;color:#94a3b8; }}
  .info {{ background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
           padding:10px 14px;font-size:11px;color:#475569;
           margin-bottom:16px;line-height:1.9; }}
  .legend {{ display:inline-flex;align-items:center;gap:8px;
             font-size:10px;color:#64748b; }}
  .lg {{ width:12px;height:12px;border-radius:2px;display:inline-block; }}
  .ts {{ text-align:right;font-size:10px;color:#94a3b8;margin-bottom:12px; }}
  details summary::-webkit-details-marker {{ display:none; }}
</style>
</head>
<body>
<div class='wrap'>
<div class='ts'>&#128337; {now} &nbsp;·&nbsp; {filename}</div>
<div class='hdr'>
  <h1>&#128202; SLA Breach — RCA Report</h1>
  <p>Delivered · SLA Key=Yes · {total_o:,} orders · {total_r} regions · {total_s} stores</p>
</div>
<div class='kpis'>
  <div class='kpi'><div class='v' style='color:#1e293b;'>{total_r}</div><div class='l'>Total Regions</div></div>
  <div class='kpi'><div class='v' style='color:#1e293b;'>{total_s}</div><div class='l'>Total Stores</div></div>
  <div class='kpi'><div class='v' style='color:#1e293b;'>{total_o:,}</div><div class='l'>Orders</div></div>
  <div class='kpi'><div class='v' style='color:#dc2626;'>{n_high}</div><div class='l'>HIGH Priority</div></div>
  <div class='kpi'><div class='v' style='color:#ea580c;'>{n_medium}</div><div class='l'>MEDIUM Priority</div></div>
  <div class='kpi'><div class='v' style='color:#16a34a;'>{n_ok}</div><div class='l'>Regions OK</div></div>
  <div class='kpi'><div class='v' style='color:#ea580c;'>{n_ih}</div><div class='l'>InhouseSLA Breach</div></div>
  <div class='kpi'><div class='v' style='color:#d97706;'>{n_br}</div><div class='l'>BIN→RTS Breach</div></div>
</div>
<div class='info'>
  <b>Column map:</b>&nbsp;
  OP to INP ← <code>Op to Inp 1</code> &nbsp;·&nbsp;
  INP to PK ← <code>Inp to Pk 1</code> &nbsp;·&nbsp;
  PK to BIN ← <code>Pk to Bin 1</code> &nbsp;·&nbsp;
  <b>InhouseSLA = avg(OP)+avg(INP)+avg(PK)</b> &nbsp;·&nbsp;
  BIN to RTS ← <code>Bin to Rts 1</code>
  <br>
  <b>Priority:</b>&nbsp;
  HIGH = InhouseSLA+BIN→RTS both breach &nbsp;·&nbsp;
  MEDIUM = either breaches &nbsp;·&nbsp; OK regions &amp; stores hidden in drill-down
  <br>
  <b>Sort:</b>&nbsp;HIGH first → InhouseSLA ↓ within HIGH &nbsp;·&nbsp;
  MEDIUM next → InhouseSLA ↓ within MEDIUM
  <br>
  <b>Region bar:</b>
  <span style='color:#16a34a;font-weight:700;'>■ OK stores</span> &nbsp;|&nbsp;
  <span style='color:#dc2626;font-weight:700;'>■ Breach stores</span> &nbsp;·&nbsp;
  <b>Store bar:</b>
  <span style='color:#16a34a;font-weight:700;'>■ OK orders</span> &nbsp;|&nbsp;
  <span style='color:#dc2626;font-weight:700;'>■ Breach orders</span>
</div>
<div class='sec-hdr'>
  <span class='title'>&#128203; Summary Report</span>
  <span class='sub'>All regions · HIGH → MEDIUM → OK · InhouseSLA ↓ within each group</span>
</div>
{summary_report_html(region_df, raw_df)}
<div class='sec-hdr'>
  <span class='title'>&#128205; Regionwise Breakdown</span>
  <span class='sub'>
    Breaching regions only · HIGH → MEDIUM · InhouseSLA ↓ · Click region → store details
  </span>
  <div class='legend' style='margin-left:auto;'>
    <span class='lg' style='background:#16a34a;'></span>OK &nbsp;&nbsp;
    <span class='lg' style='background:#dc2626;'></span>Breach
  </div>
</div>
{regionwise_breakdown_html(region_df, store_df, raw_df)}
<br><br>
</div>
</body>
</html>"""
    return html

# ───────────────────────────────────────────────────────────────────
#  STREAMLIT UI
# ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#1e293b,#334155);
            border-radius:12px;padding:22px 28px;color:#fff;
            font-family:Segoe UI,Arial,sans-serif;margin-bottom:24px;'>
  <h2 style='margin:0 0 6px;font-size:22px;letter-spacing:0.5px;color:#fff;'>
    📊 SLA Breach — RCA Report Generator
  </h2>
  <p style='margin:0;opacity:0.75;font-size:13px;'>
    Upload your Order Dump Excel file below · Report generates automatically
  </p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📁 Upload Order Dump (Excel)",
    type=['xlsx','xls'],
    help="File must have Sheet1 with OTR Status, SLA Key and stage columns"
)

if uploaded_file is not None:
    with st.spinner("⏳ Processing data and building report..."):
        try:
            file_bytes = uploaded_file.read()
            filename   = uploaded_file.name

            raw_df    = load_dump(file_bytes)
            store_df  = aggregate_by_store(raw_df)
            store_df  = flag_breaches(store_df)
            region_df = aggregate_by_region(raw_df)
            region_df = flag_breaches(region_df)

            html_report = build_report(region_df, store_df, raw_df, filename)

            st.success(f"✅ Report generated — {len(raw_df):,} orders · "
                       f"{raw_df['Store'].nunique()} stores · "
                       f"{raw_df['Region'].nunique()} regions")

            # Download button
            st.download_button(
                label="⬇️ Download Report as HTML",
                data=html_report.encode('utf-8'),
                file_name=f"SLA_RCA_Report_{datetime.now().strftime('%d%b%Y_%H%M')}.html",
                mime="text/html",
                use_container_width=True
            )

            st.markdown("---")

            # Display report inline
            components.html(html_report, height=8000, scrolling=True)

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.exception(e)
else:
    st.info("👆 Upload your Excel dump file above to generate the report")
    st.markdown("""
    <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
                padding:16px 20px;font-size:12px;color:#475569;line-height:1.9;'>
      <b>What this report generates:</b><br>
      📋 Summary Report — All regions with avg SLA times per stage<br>
      📍 Regionwise Breakdown — Breaching regions with store drill-down<br>
      🔴 HIGH = InhouseSLA + BIN→RTS both breach<br>
      🟠 MEDIUM = Either one breaches<br>
      ✅ OK = All within SLA<br>
      <br>
      <b>Required columns in Sheet1:</b>&nbsp;
      sa_name · OTR Status · SLA Key · Op to Inp 1 · Inp to Pk 1 ·
      Pk to Bin 1 · Bin to Rts 1
    </div>
    """, unsafe_allow_html=True)
