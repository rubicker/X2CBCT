from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "svg.fonttype": "path",
    }
)


OUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "runs"
    / "paper_draft"
    / "figures"
    / "assets"
    / "module_d_formulas"
)


FORMULAS = {
    "value_projection.svg": r"$\widetilde{\mathbf{V}}=\mathbf{V}\mathbf{W}_{\mathrm{HU}}$",
    "pairwise_similarity.svg": (
        r"$\mathbf{S}=\widetilde{\mathbf{V}}\widetilde{\mathbf{V}}^{\mathsf{T}}"
        r"/\sqrt{d_{\mathrm{out}}}$"
    ),
    "rowwise_attention.svg": (
        r"$\mathbf{A}=\operatorname{softmax}_{\mathrm{row}}(\mathbf{S})$"
    ),
    "attention_aggregation.svg": (
        r"$\mathbf{Z}=\mathbf{A}\widetilde{\mathbf{V}},\quad "
        r"\hat y(P)=\frac{1}{20}\sum_{i=1}^{20}Z_i$"
    ),
    "contextual_mix.svg": r"$Z_i=\sum_j A_{ij}\,\tilde v_j$",
    "direct_scalar.svg": r"$\hat y(P)=\operatorname{mean}_{i}(Z_i)$",
    "hu_denormalization.svg": r"$\widehat{\mathrm{HU}}(P)=2350\,\hat y(P)-290$",
    "input_shape.svg": r"$\mathbf{V}(P)\in\mathbb{R}^{20\times16}$",
    "candidate_shape.svg": r"$\widetilde{\mathbf{V}}\in\mathbb{R}^{20\times1}$",
    "score_shape.svg": (
        r"$\mathbf{S}\in\mathbb{R}^{20\times20}\;\mathrm{(symmetric)}$"
    ),
    "attention_row_sum.svg": r"$\sum_j A_{ij}=1$",
    "selected_row.svg": r"$\mathbf{A}_{i}$",
    "contextual_shape.svg": r"$\mathbf{Z}\in\mathbb{R}^{20\times1}$",
    "multiply_symbol.svg": r"$\times$",
    "normalized_scalar.svg": r"$\hat y(P)$",
    "shared_weight.svg": r"$\mathbf{W}_{\mathrm{HU}}$",
    "scalar_output.svg": r"$\widehat{\mathrm{HU}}(P)$",
    "volume_domain.svg": r"$\forall\,P\in\Omega$",
}


def render_svg(text: str, output: Path) -> None:
    fig = plt.figure(figsize=(0.01, 0.01), dpi=144)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")
    ax.text(
        0.0,
        0.5,
        text,
        ha="left",
        va="center",
        fontsize=18,
        color="#121418",
    )
    fig.savefig(
        output,
        format="svg",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.015,
    )
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, formula in FORMULAS.items():
        render_svg(formula, OUT_DIR / filename)
        print(OUT_DIR / filename)


if __name__ == "__main__":
    main()
