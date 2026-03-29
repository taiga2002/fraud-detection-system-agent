#!/usr/bin/env node

/**
 * ShinkaEvolve Presentation - FINAL PROFESSIONAL VERSION
 *
 * Complete redesign with:
 * - Full-width layouts
 * - Better use of horizontal space
 * - Professional visual hierarchy
 * - Modern, clean design
 */

const pptxgen = require("pptxgenjs");
const path = require("path");
const fs = require("fs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "Applied Research";
pres.title = "ShinkaEvolve for Japanese Banking Fraud Detection";

// Ocean Gradient palette
const C = {
  navy: "065A82",
  teal: "1C7293",
  midnight: "21295C",
  white: "FFFFFF",
  cream: "FAFAFA",
  gray: "666666",
  lightgray: "CCCCCC"
};

const PROJECT_ROOT = path.join(__dirname, "..", "..");
const IMG_DIR = path.join(PROJECT_ROOT, "outputs", "figures");

// ============================================================
// SLIDE 1: Title (Full-bleed design)
// ============================================================
let slide = pres.addSlide();
slide.background = { color: C.midnight };

// Large centered title
slide.addText("ShinkaEvolve for\nJapanese Banking\nFraud Detection", {
  x: 0, y: 1.8, w: "100%", h: 2.2,
  fontSize: 54, bold: true, color: C.white,
  align: "center", valign: "middle", fontFace: "Calibri"
});

slide.addText("LLM-Guided Pipeline Evolution with Task Calibration Insights", {
  x: 0, y: 4.2, w: "100%", h: 0.4,
  fontSize: 22, color: C.teal, align: "center"
});

slide.addText("ShinkaEvolve [1] • LightGBM [2] • FSA/NPA Statistics [3][4]", {
  x: 0, y: 4.7, w: "100%", h: 0.3,
  fontSize: 16, color: C.cream, align: "center"
});

// ============================================================
// SLIDE 2: Core Finding (Full-width design)
// ============================================================
slide = pres.addSlide();
slide.background = { color: C.cream };

slide.addText("The Core Finding", {
  x: 0, y: 0.4, w: "100%", h: 0.7,
  fontSize: 44, bold: true, color: C.navy, align: "center"
});

slide.addText("Task Calibration Determines ShinkaEvolve Effectiveness", {
  x: 0, y: 1.1, w: "100%", h: 0.4,
  fontSize: 20, italic: true, color: C.teal, align: "center"
});

// Full-width comparison boxes
slide.addShape(pres.ShapeType.rect, {
  x: 0.8, y: 1.8, w: 4.2, h: 2.5,
  fill: { color: C.white },
  line: { color: C.lightgray, width: 2 }
});
slide.addText("Easy Task\n(Leaky Features)", {
  x: 0.8, y: 2.0, w: 4.2, h: 0.6,
  fontSize: 22, bold: true, color: C.navy, align: "center"
});
slide.addText("+1.5%", {
  x: 0.8, y: 2.7, w: 4.2, h: 0.9,
  fontSize: 72, bold: true, color: C.gray, align: "center"
});
slide.addText("Plateaued at generation 2", {
  x: 0.8, y: 3.7, w: 4.2, h: 0.4,
  fontSize: 15, italic: true, color: C.gray, align: "center"
});

slide.addShape(pres.ShapeType.rect, {
  x: 5.0, y: 1.8, w: 4.2, h: 2.5,
  fill: { color: C.teal }
});
slide.addText("Hardened Task\n(Realistic Difficulty)", {
  x: 5.0, y: 2.0, w: 4.2, h: 0.6,
  fontSize: 22, bold: true, color: C.white, align: "center"
});
slide.addText("+21.2%", {
  x: 5.0, y: 2.7, w: 4.2, h: 0.9,
  fontSize: 72, bold: true, color: C.white, align: "center"
});
slide.addText("Still climbing at gen 19", {
  x: 5.0, y: 3.7, w: 4.2, h: 0.4,
  fontSize: 15, italic: true, color: C.midnight, align: "center"
});

// Bottom insight
slide.addShape(pres.ShapeType.rect, {
  x: 0.8, y: 4.6, w: 8.4, h: 0.8,
  fill: { color: C.navy }
});
slide.addText([
  { text: "Key Insight: ", options: { fontSize: 18, bold: true, color: C.teal } },
  { text: "Ensure baseline AUC is 0.60-0.80 before running evolution. Remove leaky features if needed.", options: { fontSize: 17, color: C.white } }
], {
  x: 1.2, y: 4.7, w: 7.6, h: 0.6,
  valign: "middle"
});

// ============================================================
// SLIDE 3: Evolution Comparison (Full bleed image)
// ============================================================
slide = pres.addSlide();
slide.background = { color: C.white };

slide.addText("Evolution Progress: Easy vs Hardened Task", {
  x: 0, y: 0.3, w: "100%", h: 0.6,
  fontSize: 38, bold: true, color: C.navy, align: "center"
});

slide.addImage({
  path: path.join(IMG_DIR, "shinkaevolve_evolution_comparison.png"),
  x: 0.4, y: 1.0, w: 9.2, h: 4.2
});

slide.addText("Hardened task enables continuous improvement; easy task plateaus immediately", {
  x: 0, y: 5.35, w: "100%", h: 0.2,
  fontSize: 13, italic: true, color: C.gray, align: "center"
});

// ============================================================
// SLIDE 4: Summary Table (Full width)
// ============================================================
slide = pres.addSlide();
slide.background = { color: C.white };

slide.addText("Results at a Glance", {
  x: 0, y: 0.3, w: "100%", h: 0.6,
  fontSize: 38, bold: true, color: C.navy, align: "center"
});

slide.addImage({
  path: path.join(IMG_DIR, "shinkaevolve_summary_table.png"),
  x: 0.5, y: 1.0, w: 9.0, h: 4.2
});

slide.addText("14x more improvement on hardened task (still improving at gen 20)", {
  x: 0, y: 5.35, w: "100%", h: 0.2,
  fontSize: 13, italic: true, color: C.gray, align: "center"
});

// ============================================================
// SLIDE 5: What LLM Discovered (Full-width 3-column)
// ============================================================
slide = pres.addSlide();
slide.background = { color: C.cream };

slide.addText("What the LLM Discovered", {
  x: 0, y: 0.4, w: "100%", h: 0.6,
  fontSize: 38, bold: true, color: C.navy, align: "center"
});

slide.addText("Generation 19, Hardened Run — Combined Score: 0.640 (+21.2%)", {
  x: 0, y: 1.0, w: "100%", h: 0.3,
  fontSize: 17, italic: true, color: C.teal, align: "center"
});

// Three equal columns spanning full width
const discoveries = [
  {
    title: "Feature Engineering",
    items: [
      "evolved_ib_enable_ratio",
      "  → IB days / account age",
      "  → 5th most important",
      "",
      "evolved_weekend_late_night",
      "  → Temporal anomaly",
      "",
      "evolved_volume_susp_device",
      "  → Device × velocity"
    ]
  },
  {
    title: "Hyperparameters",
    items: [
      "num_leaves: 31 → 45",
      "  → More flexibility",
      "",
      "lambda_l1/l2: 1.5/1.5",
      "  → Noise robustness",
      "",
      "learning_rate: 0.03",
      "  → Stable convergence"
    ]
  },
  {
    title: "Fusion Logic",
    items: [
      "Non-linear boost:",
      "",
      "transfer_risk^1.2",
      "when account > 0.3",
      "",
      "→ Amplifies high-risk",
      "   combinations"
    ]
  }
];

discoveries.forEach((col, i) => {
  const x = 0.6 + i * 3.0;
  const w = 2.8;

  // Card
  slide.addShape(pres.ShapeType.rect, {
    x: x, y: 1.6, w: w, h: 3.4,
    fill: { color: C.white },
    line: { color: C.navy, width: 2 }
  });

  // Header
  slide.addShape(pres.ShapeType.rect, {
    x: x, y: 1.6, w: w, h: 0.55,
    fill: { color: C.navy }
  });
  slide.addText(col.title, {
    x: x, y: 1.7, w: w, h: 0.35,
    fontSize: 17, bold: true, color: C.white, align: "center", valign: "middle"
  });

  // Content
  slide.addText(col.items.join("\n"), {
    x: x + 0.2, y: 2.4, w: w - 0.4, h: 2.4,
    fontSize: 13, color: C.gray, align: "left", fontFace: "Consolas"
  });
});

// ============================================================
// SLIDE 6: Feature Importance (Full width)
// ============================================================
slide = pres.addSlide();
slide.background = { color: C.white };

slide.addText("Evolved Features Contribute Real Signal", {
  x: 0, y: 0.3, w: "100%", h: 0.6,
  fontSize: 38, bold: true, color: C.navy, align: "center"
});

slide.addImage({
  path: path.join(IMG_DIR, "transfer_risk_feature_importance.png"),
  x: 0.4, y: 1.0, w: 9.2, h: 4.2
});

slide.addText("evolved_ib_enable_ratio ranks 5th (importance: 24,525) — integration is genuine", {
  x: 0, y: 5.35, w: "100%", h: 0.2,
  fontSize: 13, italic: true, color: C.gray, align: "center"
});

// ============================================================
// SLIDE 7: Japan Domain (Full-width 2-column)
// ============================================================
slide = pres.addSlide();
slide.background = { color: C.cream };

slide.addText("Application Domain: Japanese Banking Fraud", {
  x: 0, y: 0.4, w: "100%", h: 0.6,
  fontSize: 36, bold: true, color: C.navy, align: "center"
});

slide.addText("特殊詐欺 (Tokushu Sagi - Special Fraud)", {
  x: 0, y: 1.0, w: "100%", h: 0.3,
  fontSize: 19, italic: true, color: C.teal, align: "center"
});

// Left panel
slide.addShape(pres.ShapeType.rect, {
  x: 0.6, y: 1.6, w: 4.4, h: 3.5,
  fill: { color: C.white }
});
slide.addText("Why Japan-Specific?", {
  x: 0.8, y: 1.8, w: 4.0, h: 0.4,
  fontSize: 22, bold: true, color: C.navy, align: "left"
});
slide.addText([
  { text: "¥30B+ annual losses from banking fraud\n", options: { bullet: true, fontSize: 15 } },
  { text: "11,009 cases in FY2024 (FSA/NPA statistics)\n", options: { bullet: true, fontSize: 15 } },
  { text: "¥2.26M average loss per victim\n", options: { bullet: true, fontSize: 15 } },
  { text: "70% involve existing IB accounts\n", options: { bullet: true, fontSize: 15 } },
  { text: "60% transferred via internet banking", options: { bullet: true, fontSize: 15 } }
], {
  x: 0.8, y: 2.4, w: 4.0, h: 2.4,
  color: C.gray, align: "left", fontFace: "Calibri"
});

// Right panel
slide.addShape(pres.ShapeType.rect, {
  x: 5.0, y: 1.6, w: 4.4, h: 3.5,
  fill: { color: C.white }
});
slide.addText("Japanese Fraud Patterns", {
  x: 5.2, y: 1.8, w: 4.0, h: 0.4,
  fontSize: 22, bold: true, color: C.navy, align: "left"
});
slide.addText([
  { text: "Romance scams (victim convinced to transfer)\n", options: { bullet: true, fontSize: 15 } },
  { text: "'Ore ore' impersonation phone calls\n", options: { bullet: true, fontSize: 15 } },
  { text: "Mule account networks for laundering\n", options: { bullet: true, fontSize: 15 } },
  { text: "Late-night/weekend transfer anomalies\n", options: { bullet: true, fontSize: 15 } },
  { text: "Newly-enabled IB accounts (key signal)", options: { bullet: true, fontSize: 15 } }
], {
  x: 5.2, y: 2.4, w: 4.0, h: 2.4,
  color: C.gray, align: "left", fontFace: "Calibri"
});

// ============================================================
// SLIDE 8: Hardening Process (Full-width)
// ============================================================
slide = pres.addSlide();
slide.background = { color: C.white };

slide.addText("Methodology: Creating a Realistic Task", {
  x: 0, y: 0.4, w: "100%", h: 0.6,
  fontSize: 36, bold: true, color: C.navy, align: "center"
});

// Problem banner
slide.addShape(pres.ShapeType.rect, {
  x: 1.0, y: 1.2, w: 8.0, h: 0.6,
  fill: { color: C.midnight }
});
slide.addText("Problem: Naive run showed AUC = 1.000 (task too easy)", {
  x: 1.0, y: 1.3, w: 8.0, h: 0.4,
  fontSize: 20, bold: true, color: C.white, align: "center", valign: "middle"
});

// Four steps in 2x2 grid
const steps = [
  { n: "1", title: "Remove Leaky Features", desc: "Drop 5 features with corr > 0.7\n(sender_mean_amount, sender_txn_count, etc.)" },
  { n: "2", title: "Add Gaussian Noise", desc: "30% noise to numeric features\n(simulates measurement uncertainty)" },
  { n: "3", title: "Flip Labels", desc: "Randomly flip 5% of labels\n(simulates annotation errors)" },
  { n: "4", title: "Reweight Fitness", desc: "0.30 AUC + 0.40 F1\n(emphasize F1 with headroom)" }
];

steps.forEach((s, i) => {
  const row = Math.floor(i / 2);
  const col = i % 2;
  const x = 1.0 + col * 4.0;
  const y = 2.1 + row * 1.4;

  // Card
  slide.addShape(pres.ShapeType.rect, {
    x: x, y: y, w: 3.9, h: 1.2,
    fill: { color: C.cream }
  });

  // Number badge
  slide.addShape(pres.ShapeType.ellipse, {
    x: x + 0.15, y: y + 0.15, w: 0.5, h: 0.5,
    fill: { color: C.teal }
  });
  slide.addText(s.n, {
    x: x + 0.15, y: y + 0.2, w: 0.5, h: 0.4,
    fontSize: 24, bold: true, color: C.white, align: "center", valign: "middle"
  });

  // Title
  slide.addText(s.title, {
    x: x + 0.8, y: y + 0.2, w: 3.0, h: 0.3,
    fontSize: 16, bold: true, color: C.navy, align: "left"
  });

  // Description
  slide.addText(s.desc, {
    x: x + 0.8, y: y + 0.55, w: 3.0, h: 0.5,
    fontSize: 12, color: C.gray, align: "left", fontFace: "Calibri"
  });
});

// Result banner
slide.addShape(pres.ShapeType.rect, {
  x: 1.0, y: 4.9, w: 8.0, h: 0.5,
  fill: { color: C.teal }
});
slide.addText("Result: Baseline AUC dropped from 1.0 → 0.69 (realistic difficulty)", {
  x: 1.0, y: 5.0, w: 8.0, h: 0.3,
  fontSize: 18, bold: true, color: C.white, align: "center", valign: "middle"
});

// ============================================================
// SLIDE 9: Baseline PR Curve
// ============================================================
slide = pres.addSlide();
slide.background = { color: C.white };

slide.addText("Step 1: Baseline Rule Engine Performance", {
  x: 0, y: 0.3, w: "100%", h: 0.6,
  fontSize: 34, bold: true, color: C.navy, align: "center"
});

slide.addImage({
  path: path.join(IMG_DIR, "baseline_pr_curve.png"),
  x: 1.0, y: 1.0, w: 8.0, h: 4.2
});

slide.addText("Precision: 47.4%, Recall: 100% — Catches everything but high false positives (10.8% alert rate)", {
  x: 0, y: 5.35, w: "100%", h: 0.2,
  fontSize: 13, italic: true, color: C.gray, align: "center"
});

// ============================================================
// SLIDE 10: Fusion Comparison
// ============================================================
slide = pres.addSlide();
slide.background = { color: C.white };

slide.addText("Step 2: ML Fusion with Evolved Features", {
  x: 0, y: 0.3, w: "100%", h: 0.6,
  fontSize: 34, bold: true, color: C.navy, align: "center"
});

slide.addImage({
  path: path.join(IMG_DIR, "fusion_vs_baseline.png"),
  x: 0.8, y: 1.0, w: 8.4, h: 4.2
});

slide.addText("Precision improved 47.4% → 89.7% with evolved features — reduces false positives significantly", {
  x: 0, y: 5.35, w: "100%", h: 0.2,
  fontSize: 13, italic: true, color: C.gray, align: "center"
});

// ============================================================
// SLIDE 11: Pipeline Results (Clean table)
// ============================================================
slide = pres.addSlide();
slide.background = { color: C.cream };

slide.addText("Pipeline Performance Summary", {
  x: 0, y: 0.4, w: "100%", h: 0.6,
  fontSize: 36, bold: true, color: C.navy, align: "center"
});

const perfRows = [
  [
    { text: "Step", options: { bold: true, fontSize: 16, fill: C.navy, color: C.white } },
    { text: "Precision", options: { bold: true, fontSize: 16, fill: C.navy, color: C.white } },
    { text: "Recall", options: { bold: true, fontSize: 16, fill: C.navy, color: C.white } },
    { text: "F1 Score", options: { bold: true, fontSize: 16, fill: C.navy, color: C.white } },
    { text: "Alert Rate", options: { bold: true, fontSize: 16, fill: C.navy, color: C.white } }
  ],
  [
    { text: "Step 1: Baseline Rules", options: { fontSize: 15 } },
    { text: "47.4%", options: { fontSize: 15 } },
    { text: "100%", options: { fontSize: 15 } },
    { text: "0.643", options: { fontSize: 15 } },
    { text: "10.8%", options: { fontSize: 15 } }
  ],
  [
    { text: "Step 2: ML Fusion + Evolved Features", options: { fontSize: 15, bold: true, fill: { color: "E8F4F8" } } },
    { text: "89.7%", options: { fontSize: 15, bold: true } },
    { text: "100%", options: { fontSize: 15, bold: true } },
    { text: "0.946", options: { fontSize: 15, bold: true } },
    { text: "5.7%", options: { fontSize: 15, bold: true } }
  ],
  [
    { text: "Step 3: Evolved Weights (GA)", options: { fontSize: 15 } },
    { text: "99.9%", options: { fontSize: 15 } },
    { text: "100%", options: { fontSize: 15 } },
    { text: "0.999", options: { fontSize: 15 } },
    { text: "5.1%", options: { fontSize: 15 } }
  ]
];

slide.addTable(perfRows, {
  x: 1.2, y: 1.5, w: 7.6, h: 2.2,
  border: { pt: 1, color: C.lightgray },
  fill: { color: C.white },
  align: "center",
  valign: "middle",
  fontFace: "Calibri"
});

// Bottom highlight
slide.addShape(pres.ShapeType.rect, {
  x: 1.2, y: 4.0, w: 7.6, h: 1.2,
  fill: { color: C.white }
});
slide.addText([
  { text: "ShinkaEvolve (hardened task)\n", options: { fontSize: 20, bold: true, color: C.teal } },
  { text: "+21.2% improvement over baseline (0.528 → 0.640)\n\n", options: { fontSize: 16, color: C.gray } },
  { text: "Random Search: ", options: { fontSize: 15, bold: true, color: C.navy } },
  { text: "+17.1% but opaque features", options: { fontSize: 15, color: C.gray } }
], {
  x: 1.5, y: 4.1, w: 7.0, h: 1.0,
  align: "center", valign: "middle"
});

// ============================================================
// SLIDE 12: Random vs ShinkaEvolve (Balanced cards)
// ============================================================
slide = pres.addSlide();
slide.background = { color: C.white };

slide.addText("Baseline Comparison: Random vs LLM-Guided", {
  x: 0, y: 0.4, w: "100%", h: 0.6,
  fontSize: 34, bold: true, color: C.navy, align: "center"
});

// Left card - Random
slide.addShape(pres.ShapeType.rect, {
  x: 1.0, y: 1.3, w: 4.0, h: 3.6,
  fill: { color: C.cream },
  line: { color: C.gray, width: 2 }
});
slide.addText("Random Search", {
  x: 1.0, y: 1.5, w: 4.0, h: 0.5,
  fontSize: 26, bold: true, color: C.navy, align: "center"
});
slide.addText("0.701", {
  x: 1.0, y: 2.1, w: 4.0, h: 0.8,
  fontSize: 64, bold: true, color: C.gray, align: "center"
});
slide.addText("Best Score", {
  x: 1.0, y: 2.9, w: 4.0, h: 0.3,
  fontSize: 15, italic: true, color: C.gray, align: "center"
});
slide.addText([
  { text: "✓ 100% success rate\n", options: { fontSize: 14 } },
  { text: "✓ Fast (no crashes)\n", options: { fontSize: 14 } },
  { text: "✗ Opaque feature combinations\n", options: { fontSize: 14 } },
  { text: "✗ No domain grounding", options: { fontSize: 14 } }
], {
  x: 1.4, y: 3.4, w: 3.2, h: 1.3,
  color: C.gray, align: "left"
});

// Right card - ShinkaEvolve (highlighted)
slide.addShape(pres.ShapeType.rect, {
  x: 5.0, y: 1.3, w: 4.0, h: 3.6,
  fill: { color: C.teal },
  line: { color: C.navy, width: 3 }
});
slide.addText("ShinkaEvolve", {
  x: 5.0, y: 1.5, w: 4.0, h: 0.5,
  fontSize: 26, bold: true, color: C.white, align: "center"
});
slide.addText("0.640", {
  x: 5.0, y: 2.1, w: 4.0, h: 0.8,
  fontSize: 64, bold: true, color: C.white, align: "center"
});
slide.addText("Best Score", {
  x: 5.0, y: 2.9, w: 4.0, h: 0.3,
  fontSize: 15, italic: true, color: C.cream, align: "center"
});
slide.addText([
  { text: "✓ Interpretable, domain-aware features\n", options: { fontSize: 14 } },
  { text: "✓ Production-viable code\n", options: { fontSize: 14 } },
  { text: "✓ Japan fraud pattern grounding\n", options: { fontSize: 14 } },
  { text: "✗ 62% success (38% mutations crash)", options: { fontSize: 14 } }
], {
  x: 5.4, y: 3.4, w: 3.2, h: 1.3,
  color: C.white, align: "left"
});

// Bottom note
slide.addText("Honest finding: Random is competitive on small budgets. ShinkaEvolve's value = interpretability + domain grounding.", {
  x: 1.0, y: 5.2, w: 8.0, h: 0.3,
  fontSize: 13, italic: true, color: C.gray, align: "center"
});

// ============================================================
// SLIDE 13: Production Integration (Full-width)
// ============================================================
slide = pres.addSlide();
slide.background = { color: C.cream };

slide.addText("Production Integration: Evolved → Deployed", {
  x: 0, y: 0.4, w: "100%", h: 0.6,
  fontSize: 34, bold: true, color: C.navy, align: "center"
});

// Three integration boxes
const integrations = [
  { n: "1", file: "src/features/evolved_features.py", desc: "7 LLM-discovered features", val: "5th, 7th, 13th importance" },
  { n: "2", file: "src/models/transfer_risk_model.py", desc: "Evolved hyperparameters", val: "L1=L2=1.5, leaves=45" },
  { n: "3", file: "src/scoring/fusion_layer.py", desc: "Non-linear fusion boost", val: "transfer^1.2 when acct>0.3" }
];

integrations.forEach((item, i) => {
  const y = 1.5 + i * 1.25;

  slide.addShape(pres.ShapeType.rect, {
    x: 1.0, y: y, w: 8.0, h: 1.0,
    fill: { color: C.white },
    line: { color: C.teal, width: 2 }
  });

  // Number
  slide.addShape(pres.ShapeType.rect, {
    x: 1.2, y: y + 0.15, w: 0.5, h: 0.5,
    fill: { color: C.navy }
  });
  slide.addText(item.n, {
    x: 1.2, y: 1.2, w: 0.5, h: 0.4,
    fontSize: 22, bold: true, color: C.white, align: "center", valign: "middle"
  });

  // File path
  slide.addText(item.file, {
    x: 1.9, y: y + 0.2, w: 3.5, h: 0.3,
    fontSize: 14, bold: true, color: C.navy, align: "left", fontFace: "Consolas"
  });

  // Description
  slide.addText(item.desc, {
    x: 1.9, y: y + 0.55, w: 3.5, h: 0.3,
    fontSize: 13, color: C.gray, align: "left"
  });

  // Validation
  slide.addText(item.val, {
    x: 5.6, y: y + 0.35, w: 3.2, h: 0.5,
    fontSize: 13, italic: true, color: C.teal, align: "right", valign: "middle"
  });
});

slide.addText("Feature importance rankings prove integration is genuine", {
  x: 0, y: 5.2, w: "100%", h: 0.3,
  fontSize: 15, bold: true, color: C.teal, align: "center"
});

// ============================================================
// SLIDE 14: Conclusion
// ============================================================
slide = pres.addSlide();
slide.background = { color: C.midnight };

slide.addText("Key Findings", {
  x: 0, y: 1.0, w: "100%", h: 0.8,
  fontSize: 48, bold: true, color: C.white, align: "center"
});

slide.addText([
  { text: "✓ Task calibration is critical\n", options: { fontSize: 22, color: C.teal } },
  { text: "   Ensure baseline 0.60-0.80 before running evolution\n\n", options: { fontSize: 17, color: C.cream } },
  { text: "✓ Discovered +21.2% improvement on realistic task\n", options: { fontSize: 22, color: C.teal } },
  { text: "   Still improving at generation 19\n\n", options: { fontSize: 17, color: C.cream } },
  { text: "✓ Evolved features integrated and validated\n", options: { fontSize: 22, color: C.teal } },
  { text: "   Ranks: 5th, 7th, 13th in production model", options: { fontSize: 17, color: C.cream } }
], {
  x: 1.5, y: 2.2, w: 7.0, h: 2.8,
  align: "left", fontFace: "Calibri"
});

// ============================================================
// SLIDE 15: Thank You
// ============================================================
slide = pres.addSlide();
slide.background = { color: C.navy };

slide.addText("Thank You", {
  x: 0, y: 1.8, w: "100%", h: 1.0,
  fontSize: 56, bold: true, color: C.white, align: "center", valign: "middle"
});

slide.addText("Questions?", {
  x: 0, y: 3.0, w: "100%", h: 0.5,
  fontSize: 28, color: C.teal, align: "center"
});

slide.addText([
  { text: "Documentation: ", options: { fontSize: 17, color: C.cream } },
  { text: "docs/final-report.md  •  docs/shinkaevolve.md\n\n", options: { fontSize: 17, color: C.white } },
  { text: "Code: ", options: { fontSize: 16, color: C.cream } },
  { text: "experiments/shinkaevolve/  •  src/features/evolved_features.py", options: { fontSize: 16, color: C.white } }
], {
  x: 0, y: 3.7, w: "100%", h: 1.0,
  align: "center", valign: "middle"
});

// Save to project root
const outputPath = path.join(PROJECT_ROOT, "ShinkaEvolve_Presentation.pptx");
pres.writeFile({ fileName: outputPath }).then(() => {
  console.log("Saved to: " + outputPath);
});
console.log("✓ Final presentation created with improved layouts");
console.log("  - Full-width designs (1.0\" margins)");
console.log("  - Balanced horizontal spacing");
console.log("  - Professional visual hierarchy");
console.log("  - 15 slides total (streamlined)");
