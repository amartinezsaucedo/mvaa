import pickle

from mvaa.utils.graph import read_graphml
from mvaa.alignment.alignment import MultiViewAlignments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parents[3])

G_d = read_graphml("monoliths/jpetstore/vista_disenio_jpetstore.graphml")
G_t = read_graphml("monoliths/jpetstore/vista_datos_jpetstore.graphml")

with open("monoliths/jpetstore/alignment_results.pkl", "rb") as f:
    results = pickle.load(f)

BC_COL   = {'catalog': '#4C72B0', 'cart': '#DD8452', 'order': '#55A868', 'account': '#C44E52'}
BC_LIGHT = {'catalog': '#D9E4F5', 'cart': '#FAE5D3', 'order': '#D5EFD8', 'account': '#F5D5D5'}
NEUTRAL_COL   = '#888888'   # SUPPLIER: no BC defined
NEUTRAL_LIGHT = '#EBEBEB'
TEXT_DARK = '#1A1A2E'
PANEL_BG  = '#F7F8FA'

domain_nodes = [
    ('category',       'catalog'),  # 0
    ('item',           'catalog'),  # 1
    ('shopping cart',  'cart'),     # 2
    ('order',          'order'),    # 3
    ('account info',   'account'),  # 4
    ('authentication', 'account'),  # 5
]

impl_nodes = [
    ('CategoryMapper', 'catalog', [0.901, 0.000, 0.021, 0.057]),  # 0 strongly catalog
    ('ItemMapper',     'catalog', [0.649, 0.313, 0.000, 0.019]),  # 1 catalog + cart signal
    ('CartActionBean', 'cart',    [0.397, 0.559, 0.023, 0.000]),  # 2 cart-dominant but mixed
    ('OrderMapper',    'order',   [0.022, 0.000, 0.892, 0.067]),  # 3 strongly order
    ('LineItemMapper', 'catalog', [0.594, 0.035, 0.000, 0.343]),  # 4 cross-BC catalog/account
    ('AccountMapper',  'account', [0.027, 0.000, 0.052, 0.900]),  # 5 strongly account
]

data_nodes = [
    ('CATEGORY',    'catalog'),  # 0
    ('INVENTORY',   'catalog'),  # 1
    ('ITEM',        'catalog'),  # 2
    ('SUPPLIER',    None),       # 3  no BC — D->T only (shopping cart#0 -> SUPPLIER)
    ('ORDERS',      'order'),    # 4
    ('ORDERSTATUS', 'order'),    # 5
    ('LINEITEM',    'order'),    # 6
    ('SIGNON',      'account'),  # 7
    ('ACCOUNT',     'account'),  # 8
]

d_to_i = [
    (0, 0, 0.810),  # category      -> CategoryMapper   (raw=0.673)
    (1, 1, 0.457),  # item          -> ItemMapper        (raw=0.379)
    (2, 2, 0.128),  # shopping cart -> CartActionBean    (raw=0.106)
    (3, 3, 0.595),  # order         -> OrderMapper       (raw=0.494)
    (4, 5, 1.000),  # account info  -> AccountMapper     (raw=0.831)
    (5, 5, 0.968),  # authentication-> AccountMapper     (raw=0.804)
]

d_to_t = [
    (2, 3, 1.000),  # shopping cart -> SUPPLIER (w=1.0 from P_D_T)
]

i_to_t = [
    (0, 0, 1.000),  # CategoryMapper -> CATEGORY
    (1, 1, 0.500),  # ItemMapper     -> INVENTORY
    (1, 2, 0.500),  # ItemMapper     -> ITEM
    # CartActionBean: no DB access
    (3, 5, 0.861),  # OrderMapper    -> ORDERSTATUS
    (3, 4, 0.139),  # OrderMapper    -> ORDERS
    (4, 6, 1.000),  # LineItemMapper -> LINEITEM
    (5, 7, 0.500),  # AccountMapper  -> SIGNON
    (5, 8, 0.500),  # AccountMapper  -> ACCOUNT
]

PX       = {'domain': 0.165, 'impl': 0.490, 'data': 0.820}
NODE_W_D = 0.150
NODE_W_I = 0.176
NODE_W_T = 0.150
NODE_H   = 0.058
PIE_R    = 0.021
bc_order = ['catalog', 'cart', 'order', 'account']

def evenly_spaced(n, y_top=0.87, y_bot=0.10):
    if n == 1: return [(y_top + y_bot) / 2]
    return [y_top - i * (y_top - y_bot) / (n - 1) for i in range(n)]

yd = {i: y for i, y in enumerate(evenly_spaced(len(domain_nodes)))}
yi = {i: y for i, y in enumerate(evenly_spaced(len(impl_nodes)))}
yt = {i: y for i, y in enumerate(evenly_spaced(len(data_nodes)))}

fig = plt.figure(figsize=(15, 9.5), facecolor='white')
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

panel_widths = {'domain': 0.148, 'impl': 0.148, 'data': 0.148}
for key, cx in PX.items():
    hw = panel_widths[key]
    rect = FancyBboxPatch(
        (cx - hw, 0.04), hw * 2, 0.93,
        boxstyle='round,pad=0.008',
        facecolor=PANEL_BG, edgecolor='#D0D0D0',
        linewidth=1.0, zorder=0)
    ax.add_patch(rect)

for x in [0.320, 0.645]:
    ax.plot([x, x], [0.045, 0.970], color='#D8D8D8', lw=0.8, zorder=0, ls='--')

for cx, (title, sub) in {
    PX['domain']: ('Domain View',         '(from requirements)'),
    PX['impl']:   ('Implementation View', '(source code)'),
    PX['data']:   ('Data View',           '(DB schema)'),
}.items():
    ax.text(cx, 0.975, title, ha='center', va='top',
            fontsize=20, fontweight='bold', color=TEXT_DARK)
    ax.text(cx, 0.938, sub, ha='center', va='top',
            fontsize=15, color='#666666', style='italic')

def draw_arrow(ax, x0, y0, x1, y1, col, w, rad=0.0, dashed=False):
    alpha = 0.28 + 0.62 * w
    lw    = 0.7  + 2.0  * w
    ls    = (0, (4, 2)) if dashed else 'solid'
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=col, lw=lw, alpha=alpha,
                                connectionstyle=f'arc3,rad={rad}',
                                linestyle=ls),
                zorder=1)

for (di, ii, w) in d_to_i:
    bc  = domain_nodes[di][1]
    rad = 0.10 if (di == 5 and ii == 5) else 0.0
    draw_arrow(ax,
               PX['domain'] + NODE_W_D / 2, yd[di],
               PX['impl']   - NODE_W_I / 2, yi[ii],
               BC_COL[bc], w, rad=rad)

for (di, ti, w) in d_to_t:
    bc = domain_nodes[di][1]
    draw_arrow(ax,
               PX['domain'] + NODE_W_D / 2, yd[di],
               PX['data']   - NODE_W_T / 2, yt[ti],
               BC_COL[bc], w, rad=-0.15, dashed=True)

for (ii, ti, w) in i_to_t:
    bc = impl_nodes[ii][1]
    draw_arrow(ax,
               PX['impl'] + NODE_W_I / 2, yi[ii],
               PX['data'] - NODE_W_T / 2, yt[ti],
               BC_COL[bc], w)

def draw_box(ax, cx, y, label, bc, node_w, fontsize=15):
    fc = BC_LIGHT[bc] if bc else NEUTRAL_LIGHT
    ec = BC_COL[bc]   if bc else NEUTRAL_COL
    rect = FancyBboxPatch(
        (cx - node_w / 2, y - NODE_H / 2), node_w, NODE_H,
        boxstyle='round,pad=0.005',
        facecolor=fc, edgecolor=ec,
        linewidth=1.4, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, y, label, ha='center', va='center',
            fontsize=fontsize, color=TEXT_DARK, zorder=4,
            fontfamily='monospace', fontweight='medium')

for i, (label, bc) in enumerate(domain_nodes):
    draw_box(ax, PX['domain'], yd[i], label, bc, NODE_W_D)

for i, (label, bc) in enumerate(data_nodes):
    draw_box(ax, PX['data'], yt[i], label, bc, NODE_W_T)

for i, (label, dom_bc, dist) in enumerate(impl_nodes):
    cx, y = PX['impl'], yi[i]
    fc = BC_LIGHT[dom_bc]
    ec = BC_COL[dom_bc]
    rect = FancyBboxPatch(
        (cx - NODE_W_I / 2, y - NODE_H / 2), NODE_W_I, NODE_H,
        boxstyle='round,pad=0.005',
        facecolor=fc, edgecolor=ec,
        linewidth=1.4, zorder=3)
    ax.add_patch(rect)

    pie_cx   = cx + NODE_W_I / 2 - PIE_R - 0.006
    label_cx = (cx - NODE_W_I / 2 + pie_cx - PIE_R) / 2
    ax.text(label_cx, y, label, ha='center', va='center',
            fontsize=15, color=TEXT_DARK, zorder=4,
            fontfamily='monospace', fontweight='medium')

    colors = [BC_COL[b] for b in bc_order]
    start  = 90.0
    for val, col in zip(dist, colors):
        angle = val * 360.0
        if angle > 0.8:
            wedge = mpatches.Wedge(
                center=(pie_cx, y), r=PIE_R * 0.92,
                theta1=start - angle, theta2=start,
                facecolor=col, edgecolor='white', linewidth=0.5,
                zorder=5, transform=ax.transData)
            ax.add_patch(wedge)
        start -= angle

    circle = plt.Circle((pie_cx, y), PIE_R * 0.92,
                        fill=False, edgecolor='#AAAAAA', linewidth=0.5,
                        zorder=6, transform=ax.transData)
    ax.add_patch(circle)

bc_label_x = PX['domain'] - panel_widths['domain'] - 0.005
for bc in bc_order:
    idxs = [i for i, (_, b) in enumerate(domain_nodes) if b == bc]
    if not idxs: continue
    yc   = np.mean([yd[i] for i in idxs])
    ymin = min(yd[i] for i in idxs) - NODE_H / 2 - 0.012
    ymax = max(yd[i] for i in idxs) + NODE_H / 2 + 0.012
    bar_x = bc_label_x + 0.010
    ax.plot([bar_x, bar_x], [ymin, ymax],
            color=BC_COL[bc], lw=4, solid_capstyle='round', zorder=5)
    ax.text(bc_label_x - 0.002, yc, bc.capitalize(),
            ha='right', va='center', fontsize=18,
            color=BC_COL[bc], fontweight='bold')

ax.annotate(
    'cross-BC\n(catalog/account)',
    xy=(PX['impl'] + NODE_W_I/2 - PIE_R*0.5, 0.254),
    xytext=(0.670, 0.305),
    fontsize=15, color='#555555', style='italic', ha='center',
    arrowprops=dict(arrowstyle='->', color='#BBBBBB', lw=0.8,
                    connectionstyle='arc3,rad=-0.15'))

ax.annotate(
    'strong alignment\n(w\u2009=\u20090.83)',
    xy=(0.320, 0.177),
    xytext=(0.630, 0.148),
    fontsize=15, color='#555555', style='italic', ha='center',
    arrowprops=dict(arrowstyle='->', color='#BBBBBB', lw=0.8,
                    connectionstyle='arc3,rad=0.0'))

ax.annotate(
    'weak alignment\n(w\u2009=\u20090.13)',
    xy=(PX['domain'] + NODE_W_D/2 + 0.02, 0.562),
    xytext=(0.308, 0.648),
    fontsize=15, color='#555555', style='italic', ha='center',
    arrowprops=dict(arrowstyle='->', color='#BBBBBB', lw=0.8,
                    connectionstyle='arc3,rad=-0.12'))

ax.text(0.490, 0.490, 'D→T alignment',
        ha='center', va='center', fontsize=14,
        color='#888888', style='italic',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1))

bc_patches = [
    mpatches.Patch(facecolor=BC_LIGHT[bc], edgecolor=BC_COL[bc],
                   linewidth=1.2, label=bc.capitalize())
    for bc in bc_order
]
no_bc_patch = mpatches.Patch(facecolor=NEUTRAL_LIGHT, edgecolor=NEUTRAL_COL,
                             linewidth=1.2, label='No BC (D→T only)')
strong_line = plt.Line2D([0], [0], color='#555555', lw=2.5, label='Strong alignment')
weak_line   = plt.Line2D([0], [0], color='#555555', lw=0.9, alpha=0.5, label='Weak alignment')
dt_line     = plt.Line2D([0], [0], color='#555555', lw=1.2, ls=(0,(4,2)), label='D→T alignment')

ax.legend(
    handles=bc_patches + [no_bc_patch, strong_line, weak_line, dt_line],
    loc='lower center',
    bbox_to_anchor=(0.5, 0.005),
    ncol=8,
    fontsize=15,
    frameon=True,
    framealpha=0.97,
    edgecolor='#CCCCCC',
    facecolor='white',
    handlelength=1.6,
    columnspacing=0.8,
    handletextpad=0.5,
)

plt.savefig('results/graphics/cross_view_alignment.png',
            bbox_inches='tight', dpi=300, facecolor='white')
print("Done.")
