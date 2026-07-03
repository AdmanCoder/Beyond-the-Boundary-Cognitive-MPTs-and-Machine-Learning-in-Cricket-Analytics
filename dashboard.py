import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import math

# ─── Page Config ───
st.set_page_config(
    page_title="IPL Player DNA",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Premium CSS ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --gold-primary: #f59e0b;
        --gold-light: #fbbf24;
        --gold-dark: #d97706;
        --blue-primary: #3b82f6;
        --blue-light: #60a5fa;
        --blue-dark: #2563eb;
        --bg-primary: #070b14;
        --bg-secondary: #0f1623;
        --bg-card: rgba(15, 22, 35, 0.7);
        --bg-card-hover: rgba(20, 30, 50, 0.85);
        --border-subtle: rgba(255, 215, 0, 0.08);
        --border-glow: rgba(251, 191, 36, 0.25);
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --glass-bg: rgba(15, 22, 35, 0.6);
        --glass-border: rgba(255, 255, 255, 0.06);
        --radius-lg: 20px;
        --radius-md: 14px;
        --radius-sm: 10px;
        --shadow-glow: 0 0 40px rgba(251, 191, 36, 0.06);
    }

    /* ─── Global ─── */
    .stApp {
        background: var(--bg-primary);
        font-family: 'Inter', -apple-system, sans-serif;
    }
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background:
            radial-gradient(ellipse at 20% 0%, rgba(251, 191, 36, 0.04) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 100%, rgba(59, 130, 246, 0.03) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }

    /* ─── Sidebar ─── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1220 0%, #111b2e 40%, #0e1726 100%);
        border-right: 1px solid var(--glass-border);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown li {
        color: var(--text-primary) !important;
    }
    [data-testid="stSidebar"] .stSelectbox label {
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: var(--text-muted) !important;
        font-weight: 600;
    }

    /* ─── Hero Title ─── */
    .hero-container {
        text-align: center;
        padding: 20px 0 10px 0;
        position: relative;
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.12), rgba(251, 191, 36, 0.04));
        border: 1px solid rgba(251, 191, 36, 0.2);
        border-radius: 100px;
        padding: 6px 20px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        color: var(--gold-light);
        margin-bottom: 16px;
    }
    .hero-title {
        font-size: 3.2rem;
        font-weight: 900;
        letter-spacing: -2px;
        line-height: 1.1;
        margin: 0;
        background: linear-gradient(135deg, #ffffff 0%, #fbbf24 40%, #f59e0b 60%, #d97706 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        color: var(--text-muted);
        font-size: 0.85rem;
        font-weight: 400;
        margin-top: 8px;
        letter-spacing: 0.5px;
    }
    .hero-subtitle span {
        color: var(--text-secondary);
        font-weight: 500;
    }

    /* ─── Stat Cards ─── */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 14px;
        margin: 24px 0;
    }
    .stat-card {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: var(--radius-md);
        padding: 20px 16px;
        text-align: center;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--gold-primary), transparent);
        opacity: 0;
        transition: opacity 0.35s ease;
    }
    .stat-card:hover {
        transform: translateY(-4px);
        border-color: var(--border-glow);
        box-shadow: var(--shadow-glow);
    }
    .stat-card:hover::before { opacity: 1; }

    .stat-icon {
        font-size: 1.5rem;
        margin-bottom: 6px;
        display: block;
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        background: linear-gradient(135deg, var(--gold-light), var(--gold-dark));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.2;
    }
    .stat-value.blue {
        background: linear-gradient(135deg, var(--blue-light), var(--blue-dark));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-label {
        color: var(--text-muted);
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 6px;
    }

    /* ─── Section Headers ─── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 36px 0 20px 0;
    }
    .section-line {
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, var(--border-glow), transparent);
    }
    .section-line.right {
        background: linear-gradient(90deg, transparent, var(--border-glow));
    }
    .section-title {
        color: var(--text-secondary);
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 3px;
        text-transform: uppercase;
        white-space: nowrap;
    }

    /* ─── Comparison Layout ─── */
    .compare-container {
        display: flex;
        align-items: stretch;
        gap: 20px;
        margin: 24px 0;
    }
    .player-panel {
        flex: 1;
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: var(--radius-lg);
        padding: 28px 24px;
        transition: all 0.3s ease;
    }
    .player-panel:hover {
        border-color: rgba(255, 255, 255, 0.1);
    }
    .player-panel.gold { border-top: 2px solid var(--gold-primary); }
    .player-panel.blue { border-top: 2px solid var(--blue-primary); }

    .player-name {
        font-size: 1.4rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 16px;
        letter-spacing: -0.5px;
    }
    .player-name.gold { color: var(--gold-light); }
    .player-name.blue { color: var(--blue-light); }

    .mini-stats {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
    }
    .mini-stat {
        text-align: center;
        padding: 10px 4px;
        background: rgba(0, 0, 0, 0.2);
        border-radius: var(--radius-sm);
        border: 1px solid rgba(255, 255, 255, 0.03);
    }
    .mini-stat-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    .mini-stat-label {
        font-size: 0.6rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 3px;
    }

    .vs-divider {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0 4px;
    }
    .vs-circle {
        width: 48px;
        height: 48px;
        border-radius: 50%;
        background: linear-gradient(135deg, #dc2626, #991b1b);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 900;
        font-size: 0.8rem;
        color: white;
        letter-spacing: 1px;
        box-shadow: 0 0 30px rgba(220, 38, 38, 0.25);
        flex-shrink: 0;
    }

    /* ─── Insight Cards ─── */
    .insight-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 14px;
        margin: 20px 0;
    }
    .insight-card {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: var(--radius-md);
        padding: 18px 20px;
        transition: all 0.3s ease;
    }
    .insight-card:hover {
        border-color: rgba(255, 255, 255, 0.08);
        transform: translateY(-2px);
    }
    .insight-label {
        font-size: 0.65rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
    }
    .insight-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 6px 0 2px 0;
    }
    .insight-bar {
        height: 3px;
        border-radius: 2px;
        background: rgba(255, 255, 255, 0.05);
        margin-top: 10px;
        overflow: hidden;
    }
    .insight-bar-fill {
        height: 100%;
        border-radius: 2px;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .insight-bar-fill.gold {
        background: linear-gradient(90deg, var(--gold-dark), var(--gold-light));
    }
    .insight-bar-fill.green {
        background: linear-gradient(90deg, #059669, #34d399);
    }
    .insight-bar-fill.red {
        background: linear-gradient(90deg, #dc2626, #f87171);
    }
    .insight-bar-fill.blue {
        background: linear-gradient(90deg, var(--blue-dark), var(--blue-light));
    }

    /* ─── Gold Divider ─── */
    .gold-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(251, 191, 36, 0.15), transparent);
        margin: 30px 0;
    }

    /* ─── Data Table ─── */
    .feature-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 0.85rem;
    }
    .feature-table th {
        background: rgba(251, 191, 36, 0.08);
        color: var(--text-secondary);
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        padding: 12px 16px;
        text-align: left;
        border-bottom: 1px solid var(--glass-border);
    }
    .feature-table td {
        padding: 10px 16px;
        color: var(--text-primary);
        border-bottom: 1px solid rgba(255, 255, 255, 0.03);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
    }
    .feature-table tr:hover td {
        background: rgba(251, 191, 36, 0.03);
    }
    .feature-table td:first-child {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: var(--text-secondary);
    }
    .feature-table td.winner {
        color: var(--gold-light);
        font-weight: 700;
    }
    .feature-table td.loser {
        color: var(--text-muted);
    }

    /* ─── Footer ─── */
    .footer {
        text-align: center;
        padding: 40px 0 20px 0;
        color: var(--text-muted);
        font-size: 0.7rem;
        letter-spacing: 1px;
    }

    /* ─── Hide Streamlit chrome ─── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ─── Sidebar Dataset Info ─── */
    .sidebar-meta {
        background: rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: var(--radius-sm);
        padding: 16px;
        margin-top: 16px;
    }
    .sidebar-meta-row {
        display: flex;
        justify-content: space-between;
        padding: 5px 0;
        font-size: 0.8rem;
    }
    .sidebar-meta-label {
        color: var(--text-muted);
    }
    .sidebar-meta-value {
        color: var(--text-primary);
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ─── Data Loading ───
@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv("ipl_training_data_v4.csv")
    return df


@st.cache(allow_output_mutation=True)
def get_player_profile(df, player_name):
    """Build full player profile from the training data."""
    player_rows = df[df["batter_name"] == player_name]
    if len(player_rows) == 0:
        return None

    career_cols = [
        "Hand_Eye_Pace", "Head_Stability_Spin", "Pressure_Absorb", "Chase_IQ",
        "Clutch_Index", "Hard_Hitting_Power", "Technical_Adeptness",
        "Tough_Pitch_Performance", "Consistency_Gini", "Lone_Wolf",
        "Counter_Attack", "Bowler_Reading", "Perception_Skills", "Shot_Inventory",
        "Death_Specialist_SR", "Powerplay_Specialist_SR",
    ]

    profile = player_rows[career_cols].iloc[0].to_dict()

    # Basic career stats
    profile["total_balls"] = len(player_rows)
    profile["total_matches"] = player_rows["match_id"].nunique()
    profile["total_runs"] = int(player_rows["runs_batter"].sum())
    profile["wickets_as_batter"] = int(player_rows["is_wicket"].sum())
    profile["batting_sr"] = (
        profile["total_runs"] / profile["total_balls"] * 100
        if profile["total_balls"] > 0 else 0
    )
    profile["average"] = (
        profile["total_runs"] / max(profile["wickets_as_batter"], 1)
    )
    # Boundary %
    boundaries = len(player_rows[player_rows["runs_batter"].isin([4, 6])])
    profile["boundary_pct"] = boundaries / profile["total_balls"] * 100 if profile["total_balls"] > 0 else 0
    # Dot %
    dots = len(player_rows[player_rows["runs_batter"] == 0])
    profile["dot_pct"] = dots / profile["total_balls"] * 100 if profile["total_balls"] > 0 else 0
    # Sixes
    profile["sixes"] = int(len(player_rows[player_rows["runs_batter"] == 6]))
    # Fours
    profile["fours"] = int(len(player_rows[player_rows["runs_batter"] == 4]))

    # Phase-wise SR
    for phase_name, phase_val in [("Powerplay", "Powerplay"), ("Middle", "Middle"), ("Death", "Death")]:
        phase_rows = player_rows[player_rows["field_phase"] == phase_val]
        if len(phase_rows) > 0:
            profile[f"{phase_name.lower()}_sr"] = phase_rows["runs_batter"].sum() / len(phase_rows) * 100
            profile[f"{phase_name.lower()}_balls"] = len(phase_rows)
        else:
            profile[f"{phase_name.lower()}_sr"] = 0
            profile[f"{phase_name.lower()}_balls"] = 0

    return profile


def create_radar_chart(profiles, names, colors, global_maxes=None):
    """Create a premium Plotly radar chart."""
    feature_labels = [
        "Hand-Eye (Pace)", "Head Stability (Spin)", "Pressure Absorb", "Chase IQ",
        "Clutch Index", "Hard Hitting", "Technical", "Tough Pitch",
        "Consistency", "Lone Wolf", "Counter Attack", "Bowler Reading",
        "Perception", "Shot Inventory", "Death SR", "PP SR",
    ]
    feature_keys = [
        "Hand_Eye_Pace", "Head_Stability_Spin", "Pressure_Absorb", "Chase_IQ",
        "Clutch_Index", "Hard_Hitting_Power", "Technical_Adeptness",
        "Tough_Pitch_Performance", "Consistency_Gini", "Lone_Wolf",
        "Counter_Attack", "Bowler_Reading", "Perception_Skills", "Shot_Inventory",
        "Death_Specialist_SR", "Powerplay_Specialist_SR",
    ]

    fig = go.Figure()

    for idx, (profile, name, color) in enumerate(zip(profiles, names, colors)):
        raw_values = [profile.get(k, 0) for k in feature_keys]

        # Normalize: use global max across both players for fair comparison
        if global_maxes:
            normalized = []
            for v, gmax in zip(raw_values, global_maxes):
                if gmax != 0:
                    normalized.append(max(0, v / gmax * 100))
                else:
                    normalized.append(0)
        else:
            max_val = max(abs(v) for v in raw_values) if any(raw_values) else 1
            normalized = [max(0, v / max_val * 100) for v in raw_values]

        fig.add_trace(go.Scatterpolar(
            r=normalized + [normalized[0]],
            theta=feature_labels + [feature_labels[0]],
            fill="toself",
            fillcolor=color.replace("1)", "0.08)"),
            line=dict(color=color, width=2.5, shape="spline"),
            marker=dict(size=5, color=color),
            name=name,
            hovertemplate=(
                "<b>%{theta}</b><br>"
                f"<b>{name}</b><br>"
                "Value: %{customdata:.2f}<br>"
                "<extra></extra>"
            ),
            customdata=raw_values + [raw_values[0]],
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 110],
                showticklabels=False,
                gridcolor="rgba(148, 163, 184, 0.06)",
                gridwidth=1,
            ),
            angularaxis=dict(
                gridcolor="rgba(148, 163, 184, 0.08)",
                linecolor="rgba(148, 163, 184, 0.1)",
                tickfont=dict(size=10, color="#94a3b8", family="Inter"),
            ),
        ),
        showlegend=True if len(profiles) > 1 else False,
        legend=dict(
            font=dict(color="#e2e8f0", size=13, family="Inter"),
            bgcolor="rgba(0,0,0,0)",
            x=0.5, y=-0.12, xanchor="center",
            orientation="h",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=90, r=90, t=50, b=60),
        height=520,
    )

    return fig


def create_phase_chart(profile, color_scheme="gold"):
    """Create a horizontal bar chart for phase-wise SR."""
    phases = ["Powerplay", "Middle", "Death"]
    srs = [profile.get(f"{p.lower()}_sr", 0) for p in phases]
    balls = [profile.get(f"{p.lower()}_balls", 0) for p in phases]

    if color_scheme == "gold":
        colors = ["#fbbf24", "#f59e0b", "#d97706"]
    else:
        colors = ["#60a5fa", "#3b82f6", "#2563eb"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=phases,
        x=srs,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(width=0),
            cornerradius=6,
        ),
        text=[f"{sr:.0f}" for sr in srs],
        textposition="inside",
        textfont=dict(color="white", size=13, family="JetBrains Mono"),
        hovertemplate="<b>%{y}</b><br>SR: %{x:.1f}<br>Balls: %{customdata}<extra></extra>",
        customdata=balls,
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=20, t=10, b=10),
        height=160,
        xaxis=dict(
            showgrid=False, showticklabels=False, zeroline=False, range=[0, max(srs) * 1.15 if srs else 200],
        ),
        yaxis=dict(
            tickfont=dict(color="#94a3b8", size=11, family="Inter"),
            showgrid=False,
        ),
    )
    return fig


def section_header(title):
    return f"""
    <div class="section-header">
        <div class="section-line"></div>
        <span class="section-title">{title}</span>
        <div class="section-line right"></div>
    </div>
    """


def render_insights(profile, color_class="gold"):
    """Render insight cards for key metrics."""
    # Compute percentile-like bar widths (rough heuristics)
    sr_pct = min(100, profile["batting_sr"] / 2)
    avg_pct = min(100, profile["average"] / 0.6)
    boundary_pct = min(100, profile["boundary_pct"] / 0.3)
    dot_inv_pct = max(0, 100 - profile["dot_pct"] * 2.5)  # Lower dots = better

    return f"""
    <div class="insight-grid">
        <div class="insight-card">
            <div class="insight-label">Strike Rate</div>
            <div class="insight-value">{profile['batting_sr']:.1f}</div>
            <div class="insight-bar"><div class="insight-bar-fill {color_class}" style="width:{sr_pct:.0f}%"></div></div>
        </div>
        <div class="insight-card">
            <div class="insight-label">Average</div>
            <div class="insight-value">{profile['average']:.1f}</div>
            <div class="insight-bar"><div class="insight-bar-fill green" style="width:{avg_pct:.0f}%"></div></div>
        </div>
        <div class="insight-card">
            <div class="insight-label">Boundary %</div>
            <div class="insight-value">{profile['boundary_pct']:.1f}%</div>
            <div class="insight-bar"><div class="insight-bar-fill {color_class}" style="width:{boundary_pct:.0f}%"></div></div>
        </div>
        <div class="insight-card">
            <div class="insight-label">Dot Ball %</div>
            <div class="insight-value">{profile['dot_pct']:.1f}%</div>
            <div class="insight-bar"><div class="insight-bar-fill red" style="width:{profile['dot_pct'] * 2.5:.0f}%"></div></div>
        </div>
    </div>
    """


def render_feature_table(profiles, names, compare=False):
    """Render a styled HTML comparison table."""
    feature_map = {
        "Hand Eye (Pace)": "Hand_Eye_Pace",
        "Head Stability (Spin)": "Head_Stability_Spin",
        "Pressure Absorb": "Pressure_Absorb",
        "Chase IQ": "Chase_IQ",
        "Clutch Index": "Clutch_Index",
        "Hard Hitting Power": "Hard_Hitting_Power",
        "Technical Adeptness": "Technical_Adeptness",
        "Tough Pitch": "Tough_Pitch_Performance",
        "Consistency": "Consistency_Gini",
        "Lone Wolf": "Lone_Wolf",
        "Counter Attack": "Counter_Attack",
        "Bowler Reading": "Bowler_Reading",
        "Perception Skills": "Perception_Skills",
        "Shot Inventory": "Shot_Inventory",
        "Death Specialist SR": "Death_Specialist_SR",
        "PP Specialist SR": "Powerplay_Specialist_SR",
    }

    if compare and len(profiles) == 2:
        header = f"<th>{names[0]}</th><th>{names[1]}</th>"
        rows = ""
        for label, key in feature_map.items():
            v1 = profiles[0].get(key, 0)
            v2 = profiles[1].get(key, 0)
            c1 = "winner" if v1 >= v2 else "loser"
            c2 = "winner" if v2 >= v1 else "loser"
            rows += f'<tr><td>{label}</td><td class="{c1}">{v1:.2f}</td><td class="{c2}">{v2:.2f}</td></tr>'
    else:
        header = f"<th>{names[0]}</th>"
        rows = ""
        for label, key in feature_map.items():
            v = profiles[0].get(key, 0)
            rows += f"<tr><td>{label}</td><td>{v:.2f}</td></tr>"

    return f"""
    <table class="feature-table">
        <thead><tr><th>Skill Feature</th>{header}</tr></thead>
        <tbody>{rows}</tbody>
    </table>
    """


# ─── Main App ───
def main():
    df = load_data()
    all_batters = sorted(df["batter_name"].unique().tolist())

    # ─── Hero ───
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">IPL Analytics Platform</div>
        <h1 class="hero-title">Player DNA</h1>
        <p class="hero-subtitle">16-Dimension Career Skill Profiler · <span>V4.1</span> · 278,205 Deliveries</p>
    </div>
    """, unsafe_allow_html=True)

    # ─── Sidebar ───
    with st.sidebar:
        st.markdown("### Player Search")
        st.markdown("---")

        player1 = st.selectbox(
            "PLAYER 1",
            options=all_batters,
            index=all_batters.index("V Kohli") if "V Kohli" in all_batters else 0,
            key="p1",
        )

        st.markdown("")
        compare_mode = st.checkbox("Compare two players", value=False)

        player2 = None
        if compare_mode:
            player2 = st.selectbox(
                "PLAYER 2",
                options=all_batters,
                index=all_batters.index("RG Sharma") if "RG Sharma" in all_batters else 1,
                key="p2",
            )

        st.markdown("---")

        n_matches = df["match_id"].nunique()
        n_deliveries = len(df)
        n_players = len(all_batters)
        n_features = len(df.columns)

        st.markdown(f"""
        <div class="sidebar-meta">
            <div class="sidebar-meta-row">
                <span class="sidebar-meta-label">Matches</span>
                <span class="sidebar-meta-value">{n_matches:,}</span>
            </div>
            <div class="sidebar-meta-row">
                <span class="sidebar-meta-label">Deliveries</span>
                <span class="sidebar-meta-value">{n_deliveries:,}</span>
            </div>
            <div class="sidebar-meta-row">
                <span class="sidebar-meta-label">Players</span>
                <span class="sidebar-meta-value">{n_players}</span>
            </div>
            <div class="sidebar-meta-row">
                <span class="sidebar-meta-label">Features</span>
                <span class="sidebar-meta-value">{n_features}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ─── Load profiles ───
    profile1 = get_player_profile(df, player1)
    if profile1 is None:
        st.error(f"Player '{player1}' not found in the dataset.")
        return

    # ═══════════════════════════════════════════
    # SINGLE PLAYER MODE
    # ═══════════════════════════════════════════
    if not compare_mode:
        # Stat cards
        st.markdown(f"""
        <div class="stats-grid">
            <div class="stat-card">
                <p class="stat-value">{profile1['total_matches']}</p>
                <p class="stat-label">Matches</p>
            </div>
            <div class="stat-card">
                <p class="stat-value">{profile1['total_runs']:,}</p>
                <p class="stat-label">Runs</p>
            </div>
            <div class="stat-card">
                <p class="stat-value">{profile1['batting_sr']:.1f}</p>
                <p class="stat-label">Strike Rate</p>
            </div>
            <div class="stat-card">
                <p class="stat-value">{profile1['average']:.1f}</p>
                <p class="stat-label">Average</p>
            </div>
            <div class="stat-card">
                <p class="stat-value">{profile1['sixes']}</p>
                <p class="stat-label">Sixes</p>
            </div>
            <div class="stat-card">
                <p class="stat-value">{profile1['fours']}</p>
                <p class="stat-label">Fours</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Insight cards
        st.markdown(section_header("KEY METRICS"), unsafe_allow_html=True)
        st.markdown(render_insights(profile1, "gold"), unsafe_allow_html=True)

        # Radar
        st.markdown(section_header("SKILL RADAR"), unsafe_allow_html=True)
        fig = create_radar_chart([profile1], [player1], ["rgba(251, 191, 36, 1)"])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Phase SR
        st.markdown(section_header("PHASE-WISE STRIKE RATE"), unsafe_allow_html=True)
        phase_fig = create_phase_chart(profile1, "gold")
        st.plotly_chart(phase_fig, use_container_width=True, config={"displayModeBar": False})

        # Feature table
        st.markdown(section_header("ALL 16 SKILL FEATURES"), unsafe_allow_html=True)
        st.markdown(render_feature_table([profile1], [player1]), unsafe_allow_html=True)

    # ═══════════════════════════════════════════
    # COMPARISON MODE
    # ═══════════════════════════════════════════
    else:
        profile2 = get_player_profile(df, player2)
        if profile2 is None:
            st.error(f"Player '{player2}' not found.")
            return

        # Comparison panels (pure HTML, no nested columns)
        avg1 = profile1['average']
        avg2 = profile2['average']

        st.markdown(f"""
        <div class="compare-container">
            <div class="player-panel gold">
                <div class="player-name gold">{player1}</div>
                <div class="mini-stats">
                    <div class="mini-stat">
                        <div class="mini-stat-val">{profile1['total_matches']}</div>
                        <div class="mini-stat-label">Matches</div>
                    </div>
                    <div class="mini-stat">
                        <div class="mini-stat-val">{profile1['total_runs']:,}</div>
                        <div class="mini-stat-label">Runs</div>
                    </div>
                    <div class="mini-stat">
                        <div class="mini-stat-val">{profile1['batting_sr']:.1f}</div>
                        <div class="mini-stat-label">SR</div>
                    </div>
                    <div class="mini-stat">
                        <div class="mini-stat-val">{avg1:.1f}</div>
                        <div class="mini-stat-label">Avg</div>
                    </div>
                </div>
            </div>
            <div class="vs-divider">
                <div class="vs-circle">VS</div>
            </div>
            <div class="player-panel blue">
                <div class="player-name blue">{player2}</div>
                <div class="mini-stats">
                    <div class="mini-stat">
                        <div class="mini-stat-val">{profile2['total_matches']}</div>
                        <div class="mini-stat-label">Matches</div>
                    </div>
                    <div class="mini-stat">
                        <div class="mini-stat-val">{profile2['total_runs']:,}</div>
                        <div class="mini-stat-label">Runs</div>
                    </div>
                    <div class="mini-stat">
                        <div class="mini-stat-val">{profile2['batting_sr']:.1f}</div>
                        <div class="mini-stat-label">SR</div>
                    </div>
                    <div class="mini-stat">
                        <div class="mini-stat-val">{avg2:.1f}</div>
                        <div class="mini-stat-label">Avg</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Radar — use shared scale
        st.markdown(section_header("SKILL RADAR · HEAD TO HEAD"), unsafe_allow_html=True)
        feature_keys = [
            "Hand_Eye_Pace", "Head_Stability_Spin", "Pressure_Absorb", "Chase_IQ",
            "Clutch_Index", "Hard_Hitting_Power", "Technical_Adeptness",
            "Tough_Pitch_Performance", "Consistency_Gini", "Lone_Wolf",
            "Counter_Attack", "Bowler_Reading", "Perception_Skills", "Shot_Inventory",
            "Death_Specialist_SR", "Powerplay_Specialist_SR",
        ]
        global_maxes = [
            max(abs(profile1.get(k, 0)), abs(profile2.get(k, 0)), 0.01)
            for k in feature_keys
        ]

        fig = create_radar_chart(
            [profile1, profile2], [player1, player2],
            ["rgba(251, 191, 36, 1)", "rgba(96, 165, 250, 1)"],
            global_maxes=global_maxes,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Phase SR side by side
        st.markdown(section_header("PHASE-WISE STRIKE RATE"), unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<p style='text-align:center; color:#fbbf24; font-weight:600; font-size:0.85rem;'>{player1}</p>", unsafe_allow_html=True)
            st.plotly_chart(create_phase_chart(profile1, "gold"), use_container_width=True, config={"displayModeBar": False})
        with col2:
            st.markdown(f"<p style='text-align:center; color:#60a5fa; font-weight:600; font-size:0.85rem;'>{player2}</p>", unsafe_allow_html=True)
            st.plotly_chart(create_phase_chart(profile2, "blue"), use_container_width=True, config={"displayModeBar": False})

        # Feature table with winner highlighting
        st.markdown(section_header("ALL 16 SKILL FEATURES"), unsafe_allow_html=True)
        st.markdown(
            render_feature_table([profile1, profile2], [player1, player2], compare=True),
            unsafe_allow_html=True,
        )

    # Footer
    st.markdown("""
    <div class="footer">
        IPL PLAYER DNA · V4.1 DATASET · READ-ONLY VISUALIZATION · BUILT FOR RESEARCH
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
