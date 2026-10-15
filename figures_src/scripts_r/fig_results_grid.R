# fig_results_grid.R -- Merged results composite (paper Fig. 5, 2 x 2 grid,
# 183 x 142 mm). v1 (2026-10-02): merges the former fig4_ladder panel (a),
# fig6_resource panel (a) and fig7_domain into one four-panel figure.
#   (a) certified pass-rate ladder across the four main configurations
#       (0 -> 1 -> 117 -> 191 of 218), shallow-chain and 7B reference marks;
#   (b) per-question resource consumption, log axis (tokens / time / memory);
#   (c) Lean-Workbook triage funnel (132,378 -> 48,106 -> 3,536 -> 2,132);
#   (d) nine expansion families, model-only chain vs family-tree chain.
# Data: inline judged counts; ../../results/gen37/GEN37_COMPARE.json;
# data_cited.json (GEN44_FINAL_REPORT r3 transcription). The 2x2 factorial
# panel of the former fig4_ladder is not carried: its numbers remain in
# Table 7 of the manuscript.
library(ggplot2)
library(patchwork)
source("scripts_r/pub_style_r.R")

# v2 (2026-10-03): hairline axes 0.3 mm (matched to the matplotlib track's
# 0.8 pt reference); ladder bars narrowed 0.62 half-width; in-bar white
# percentage labels moved above the bars as a second dark line (they used to
# overflow the narrower bars); resource grey-bar labels moved outside the bar
# end (white "Baseline 168" used to overflow the bar).
theme_big <- function() {
  theme_pub(base_size = 10) +
    theme(
      axis.line         = element_line(linewidth = 0.3, colour = "black"),
      axis.ticks        = element_line(linewidth = 0.3, colour = "black"),
      axis.ticks.length = unit(2.5, "pt"),
      axis.title        = element_text(size = 10, face = "bold", colour = "black"),
      axis.text         = element_text(size = 9, colour = "black"),
      plot.tag          = element_text(size = 12, face = "bold",
                                       family = "Times New Roman")
    )
}

## (a) pass-rate ladder -------------------------------------------------------
N <- 218
cols4 <- c("#D4D4D4", "#A5A5A5", PAL$blue_sec, PAL$blue_main)

p_a <- ggplot(data.frame(x = 1:4,
                         y = c(0, 1 / N, 117 / N, 191 / N),
                         lab = c("0/218", "1/218", "117/218\n53.7%",
                                 "191/218\n87.6%"),
                         col = cols4), aes(x, y)) +
  geom_rect(aes(xmin = x - 0.20, xmax = x + 0.20, ymin = 0, ymax = y, fill = I(col))) +
  geom_text(aes(label = lab), vjust = 0, nudge_y = 0.055, size = 3.3,
            fontface = "bold", family = "Times New Roman", lineheight = 0.9) +
  annotate("text", x = 0.58, y = 1.075, label = "Shallow chain 180/1083",
           hjust = 0, vjust = 1, size = 2.9, colour = PAL$grey_d) +
  annotate("text", x = 0.58, y = 0.975, label = "Baseline model 18/218",
           hjust = 0, vjust = 1, size = 2.9, colour = PAL$red_s) +
  scale_x_continuous(breaks = 1:4,
                     labels = c("Direct\ngeneration", "Slots +\ntrue-compute",
                                "Planning \u00d7\ncompiler", "State shift\n(final)"),
                     limits = c(0.45, 4.55), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 1.12), breaks = c(0, 0.25, 0.5, 0.75, 1),
                     expand = c(0, 0)) +
  labs(x = NULL, y = "Certified pass rate") +
  theme_big() +
  theme(axis.text.x = element_text(size = 7.8, colour = "black"))

## (b) resource consumption ---------------------------------------------------
C <- jsonlite::fromJSON("../data/raw/results_gen37/GEN37_COMPARE.json")
ours <- c(C$ours_135m$chain_steps_per_q, C$ours_135m$secs_per_q, 833.0)
ds <- c(C$deepseek$out_tokens_per_q, C$deepseek$secs_per_q, 8567.0)
ours_lab <- c("6.12", "0.53", "833")
ds_lab <- c("168", "19.88", "8567")
# ratios follow the displayed pair values (19.88 / 0.53 = 37.5), matching the
# manuscript's 37.5x wall-clock wording; 833 is the eval-phase memory plateau
# observed in the gen36 GPU2 meters (peak 837 MiB, file floor 11-14 MiB).
mult <- c("27\u00d7", "37.5\u00d7", "10\u00d7")
groups <- c("Tokens / question", "Time (s / question)", "Memory (MiB)")
ytop <- 2:0
h <- 0.30

bar_df <- data.frame(
  ymin = c(ytop + 0.02, ytop - h - 0.02),
  ymax = c(ytop + 0.02 + h, ytop - 0.02),
  xmin = rep(0.2, 6),
  xmax = c(ds, ours),
  col = rep(c("#9E9E9E", PAL$blue_main), each = 3))

p_b <- ggplot() +
  geom_rect(data = bar_df, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax,
                               fill = I(col))) +
  annotate("text", x = ours[1] * 1.3, y = ytop[1] - h / 2 - 0.02,
           label = "Ours 6.12 (decision actions)", hjust = 0, size = 3.2,
           fontface = "bold", colour = PAL$blue_main) +
  annotate("text", x = ds[1] * 1.12, y = ytop[1] + h / 2 + 0.02,
           label = "Baseline 168", hjust = 0, size = 3.2,
           fontface = "bold", colour = PAL$ink) +
  annotate("text", x = ds[2:3] * 1.22, y = ytop[2:3] + h / 2 + 0.02,
           label = ds_lab[2:3], hjust = 0, size = 3.2,
           fontface = "bold", colour = PAL$ink) +
  annotate("text", x = ours[2:3] * 1.25, y = ytop[2:3] - h / 2 - 0.02,
           label = ours_lab[2:3], hjust = 0, size = 3.2,
           colour = PAL$ink) +
  annotate("text", x = 2.9e4, y = ytop, label = mult, hjust = 1, size = 3.5,
           fontface = "bold", colour = PAL$blue_main) +
  scale_x_log10(limits = c(0.2, 3.1e4), breaks = c(1, 10, 100, 1000, 10000),
                labels = parse(text = c("1", "10", "100", "1000", "10^4")),
                expand = c(0, 0)) +
  scale_y_continuous(breaks = 2:0, labels = groups, limits = c(-0.55, 2.55),
                     expand = c(0, 0)) +
  labs(x = "Per-question consumption (log scale)", y = NULL) +
  theme_big() +
  theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(),
        axis.text.y = element_text(size = 8.5, colour = "black"))

## (c) Lean-Workbook triage funnel --------------------------------------------
D <- jsonlite::fromJSON("data_cited.json")
F <- D$funnel

layers <- data.frame(
  name = c("Ingested", "In-domain", "Rendered & judged", "Certified"),
  v = c(F$denominator, F$in_domain, F$judged, F$certified),
  col = c("#C9D9EA", "#8FB3D6", PAL$blue_sec, PAL$blue_main),
  note = c("7,448 library-corrupt rows removed", "84,267 out-of-domain logged",
           "not executed \u2260 scored zero", "60.29% (Wilson 58.67\u201361.89)"),
  ncol = c(PAL$grey_d, PAL$grey_d, PAL$grey_d, PAL$blue_main),
  stringsAsFactors = FALSE)
layers$y <- 3:0
DM <- F$denominator

p_c <- ggplot(layers, aes(v, y)) +
  geom_rect(aes(xmin = 0, xmax = v, ymin = y - 0.29, ymax = y + 0.29, fill = I(col))) +
  geom_text(data = subset(layers, v >= 40000),
            aes(x = v * 0.985, label = format(v, big.mark = ",")),
            hjust = 1, size = 3.4, fontface = "bold", colour = "white") +
  geom_text(data = subset(layers, v < 40000),
            aes(x = v + DM * 0.015, label = format(v, big.mark = ",")),
            hjust = 0, size = 3.4, fontface = "bold", colour = PAL$ink) +
  geom_text(aes(x = DM * 1.02, y = y - 0.36, label = note, colour = I(ncol)),
            hjust = 1, vjust = 1, size = 2.9) +
  scale_x_continuous(limits = c(0, DM * 1.04), breaks = c(0, 50000, 100000),
                     labels = c("0", "50k", "100k"), expand = c(0, 0)) +
  scale_y_continuous(breaks = 3:0, labels = layers$name,
                     limits = c(-0.75, 3.55), expand = c(0, 0)) +
  labs(x = "Rows (Lean-Workbook ingest)", y = NULL) +
  theme_big() +
  theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(),
        axis.text.y = element_text(size = 8.5, colour = "black"))

## (d) nine expansion families ------------------------------------------------
fams <- D$gen44_families
mv <- ifelse(fams$model_judged > 0, fams$model_cert / fams$model_judged, NA)
tv <- fams$tree_cert / fams$tree_judged
tot <- D$gen44_totals$model$rate
yf <- rev(seq_len(nrow(fams)))

seg <- data.frame(x = mv, xend = tv, y = yf)
seg <- seg[is.finite(seg$x), ]
p_m <- data.frame(x = mv, y = yf)
p_t <- data.frame(x = tv, y = yf)

p_d <- ggplot() +
  geom_segment(data = seg, aes(x = x, xend = xend, y = y, yend = y),
               colour = "#C9C9C9", linewidth = 0.3) +
  geom_point(data = p_m, aes(x, y, colour = "Model-only (623)"), size = 2.5) +
  geom_point(data = p_t, aes(x, y, colour = "Family-tree (1,525)"), size = 2.5) +
  geom_vline(xintercept = tot, linetype = "dotted", linewidth = 0.3,
             colour = PAL$grey_d) +
  annotate("text", x = tot + 0.015, y = 9.45,
           label = "Arm totals 45.10% / 44.07%", hjust = 0, size = 2.9,
           colour = PAL$grey_d) +
  scale_colour_manual(values = c("Model-only (623)" = PAL$blue_main,
                                 "Family-tree (1,525)" = PAL$tol_green),
                      breaks = c("Model-only (623)",
                                 "Family-tree (1,525)"),
                      name = NULL) +
  scale_x_continuous(limits = c(0, 1.08), breaks = c(0, 0.25, 0.5, 0.75, 1),
                     labels = c("0%", "25%", "50%", "75%", "100%"),
                     expand = c(0, 0)) +
  scale_y_continuous(breaks = 9:1, labels = fams$fam,
                     limits = c(0.35, 9.85), expand = c(0, 0)) +
  labs(x = "End-to-end certified pass rate", y = NULL) +
  theme_big() +
  theme(legend.position = c(0.5, 1.10), legend.justification = c(0.5, 1),
        legend.direction = "horizontal",
        legend.text = element_text(size = 8.2, colour = "black"),
        legend.key.height = unit(8, "pt"),
        plot.margin = margin(t = 26, r = 2, b = 2, l = 2, unit = "pt"))

fig <- p_a + p_b + p_c + p_d +
  plot_layout(nrow = 2, heights = c(1, 1.18)) +
  plot_annotation(tag_levels = "a")
fig <- fig & theme(plot.tag = element_text(size = 12, face = "bold",
                                           family = "Times New Roman"))
save_pub_r(fig, "output/fig_results_grid", width_mm = 183, height_mm = 142,
           panel_ids = c("a", "b", "c", "d"),
           exemptions = list(list(
             panels = "c", checks = "panel-width",
             reason = "triage axis intentionally narrower than nine-family panel (longer y labels)")))
