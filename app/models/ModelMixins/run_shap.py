import mlflow
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_shap_top10(X, shap_values, feature_names, model_type):
    # ── 1. Prepare SHAP values DataFrame ─────────────────────────────────────────
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
 
    # ── 2. Top-10 features by mean |SHAP| ───────────────────────────────────────
    mean_abs   = shap_df.abs().mean().sort_values(ascending=False)
    top10      = mean_abs.head(10).index.tolist()
    
    # Mean SHAP when follow (0) vs violate (1) for each top-10 feature
    records = []
    for feat in top10:
        mask_follow  = X[feat] == 0
        mask_violate = X[feat] == 1
        records.append({
            "feature":       feat,
            "mean_abs_shap": mean_abs[feat],
            "shap_follow":   shap_df.loc[mask_follow, feat].mean(),
            "shap_violate":  shap_df.loc[mask_violate, feat].mean(),
        })
    plot_df = pd.DataFrame(records).sort_values("mean_abs_shap", ascending=True)
    
    # ── 3. Plot ──────────────────────────────────────────────────────────────────
    COLOR_FOLLOW  = "#4C9BE8"   # cool blue  → rule followed
    COLOR_VIOLATE = "#E8624C"   # warm red   → rule violated
    COLOR_ABS     = "#B0B8C1"   # neutral grey for mean |SHAP| diamonds
    BG            = "white"
    PANEL         = "white"
    TEXT          = "black"
    GRID          = "gray"
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                            gridspec_kw={"width_ratios": [2, 1]})
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(PANEL)
    
    y_pos = np.arange(len(plot_df))
    feat_labels = plot_df["feature"].values
    
    # ── Left panel: diverging bars (follow vs violate) ───────────────────────────
    ax = axes[0]
    bar_h = 0.38
    
    bars_f = ax.barh(y_pos - bar_h / 2, plot_df["shap_follow"],
                    height=bar_h, color=COLOR_FOLLOW,  alpha=0.88,
                    label="Follow (0)", zorder=3)
    bars_v = ax.barh(y_pos + bar_h / 2, plot_df["shap_violate"],
                    height=bar_h, color=COLOR_VIOLATE, alpha=0.88,
                    label="Violate (1)", zorder=3)
    
    ax.axvline(0, color=TEXT, linewidth=0.8, alpha=0.5, zorder=4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_labels, color=TEXT, fontsize=12, fontfamily="monospace")
    ax.set_xlabel("Mean SHAP Value", color=TEXT, fontsize=12)
    ax.set_title("Mean SHAP: Follow vs Violate", color=TEXT,
                fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    
    legend = ax.legend(framealpha=0.15, facecolor=PANEL, edgecolor=GRID,
                    labelcolor=TEXT, fontsize=9, loc="lower right")
    
    # ── Right panel: mean |SHAP| dot plot ────────────────────────────────────────
    ax2 = axes[1]
    ax2.scatter(plot_df["mean_abs_shap"], y_pos,
                color=COLOR_ABS, s=90, zorder=4, edgecolors=TEXT,
                linewidths=0.5)
    for i, (val, yp) in enumerate(zip(plot_df["mean_abs_shap"], y_pos)):
        ax2.plot([0, val], [yp, yp], color=COLOR_ABS,
                linewidth=1.2, alpha=0.5, zorder=3)
        ax2.text(val + 0.007, yp, f"{val:.3f}", va="center",
                color=TEXT, fontsize=10, fontfamily="monospace")
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.set_xlabel("Mean |SHAP|", color=TEXT, fontsize=12)
    ax2.set_title("Rule Importance", color=TEXT,
                fontsize=14, fontweight="bold", pad=12)
    ax2.tick_params(colors=TEXT, labelsize=9)
    ax2.spines[["top","right","left","bottom"]].set_visible(False)
    ax2.xaxis.grid(True, color=GRID, linewidth=0.6, linestyle="--", zorder=0)
    ax2.set_axisbelow(True)
    ax2.set_xlim(left=0)
    
    # ── Super-title & tight layout ───────────────────────────────────────────────
    #fig.suptitle(f"{model_type} - SHAP Tree Explainer  -  Top-10 GOLDBAR Rules",
                #color=TEXT, fontsize=14, fontweight="bold", y=1.01)
    
    plt.tight_layout(pad=1.8)

    # save figure to mlflow artifacts
    mlflow.log_figure(fig, f"shap_top10_{model_type}_rules.png")
    plt.close(fig)