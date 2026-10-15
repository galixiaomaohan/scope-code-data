# fig3_law1.R -- Quantitative test of Law 1 (two panels + pooled inset)
# v2 (2026-10-02): legibility pass -- larger bold axis titles, larger
# annotations, thicker axes/ticks/data lines, canvas 183x85 -> 162x66 mm.
# (a) power-law decay p^k with held-out control; (b) controlled
# integer damage, unit-level survival vs independence theory 1-p, inset: pooled
# whole-proof level. Data: data_cited.json (gen39 T4_DAMAGE.json transcription).
library(ggplot2)
library(patchwork)
source("scripts_r/pub_style_r.R")
D <- jsonlite::fromJSON("data_cited.json", simplifyVector = TRUE)

wilson_ci <- function(k, n, z = 1.959964) {
  if (n == 0) return(c(0, 1))
  p <- k / n; d <- 1 + z^2 / n; c0 <- p + z^2 / (2 * n)
  h <- z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2))
  c((c0 - h) / d, (c0 + h) / d)
}

# Legibility theme: bold 10 pt titles, 9 pt ticks; hairline axes 0.3 mm
# (v3, 2026-10-03: matched to the matplotlib track's 0.8 pt reference).
# v3 also replaces the cowplot::ggdraw inset wrapper (which blocked patchwork
# panel alignment) with in-panel annotate() layers, so (a)/(b) now align.
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

# ---------------- (a) power law ----------------
curve <- data.frame(k = seq(1, 30, length.out = 300), y = 0.79^seq(1, 30, length.out = 300))
held_lohi <- wilson_ci(1, 20)
held <- data.frame(k = 14.9, y = 0.05, lo = held_lohi[1], hi = held_lohi[2])

p_a <- ggplot() +
  geom_line(data = curve, aes(k, y, colour = "Power law"), linewidth = 0.45) +
  scale_colour_manual(values = c("Power law" = PAL$blue_main),
                      labels = c("Power law" = parse(text = '"Power law"~p^k')),
                      name = NULL) +
  ggnewscale::new_scale_colour() +
  geom_errorbar(data = held, aes(x = k, ymin = lo, ymax = hi, colour = "Held-out"),
                linewidth = 0.3, width = 0.45) +
  geom_point(data = held, aes(k, y, colour = "Held-out"), size = 2.6) +
  scale_colour_manual(values = c("Held-out" = PAL$red_s),
                      labels = c("Held-out" = "Held-out 1/20"),
                      name = NULL) +
  geom_hline(yintercept = 18 / 218, linetype = "dashed", linewidth = 0.3,
             colour = PAL$grey_d) +
  annotate("text", x = 29.8, y = 0.055, label = "Baseline model",
           hjust = 1, vjust = 1, size = 3.0, colour = PAL$grey_d) +
  annotate("text", x = 29.8, y = 0.033, label = "18/218 = 8.3%",
           hjust = 1, vjust = 1, size = 3.0, colour = PAL$grey_d) +
  annotate("segment", x = 25.5, y = 0.020, xend = 25.5, yend = 0.0007,
           arrow = grid::arrow(length = unit(5, "pt"), type = "closed"),
           colour = PAL$red_s, linewidth = 0.3) +
  annotate("text", x = 24.7, y = 0.0013, label = "Deep-chain roll 0/218",
           hjust = 1, vjust = 0, size = 3.0, colour = PAL$red_s) +
  scale_y_log10(limits = c(1.5e-4, 1.5),
                breaks = c(1e-3, 1e-2, 1e-1, 1),
                labels = c("0.001", "0.01", "0.1", "1")) +
  scale_x_continuous(breaks = c(1, 5, 10, 15, 20, 25, 30), limits = c(1, 30.5)) +
  labs(x = "Number of content integers k", y = "Certified pass rate (direct)") +
  theme_big() +
  theme(legend.position = c(0.97, 0.97), legend.justification = c(1, 1),
        legend.text = element_text(size = 9, colour = "black"),
        legend.key.height = unit(10, "pt"),
        legend.spacing.y = unit(2, "pt"))

# ---------------- (b) controlled damage ----------------
cells <- D$damage_cells$cells
w <- cells$wilson
if (is.matrix(w)) { cells$lo <- as.numeric(w[, 1]); cells$hi <- as.numeric(w[, 2]) } else {
  cells$lo <- as.numeric(sapply(cells$wilson, function(v) v[1]))
  cells$hi <- as.numeric(sapply(cells$wilson, function(v) v[2]))
}
PLEV <- c("0.50", "0.70", "0.80", "0.90", "0.95", "0.99")
cells$pbin <- factor(sprintf("%.2f", cells$p), levels = PLEV)
cells$x <- as.numeric(cells$pbin)
offs <- seq(-0.21, 0.21, length.out = 4)
cells$xoff <- cells$x + offs[match(cells$bin, c("B1", "B2", "B3", "B4"))]
theory <- data.frame(x = 1:5, y = 1 - c(0.50, 0.70, 0.80, 0.90, 0.95))

p_b <- ggplot() +
  geom_line(data = theory, aes(x, y, colour = "Theory"),
            linetype = "dashed", linewidth = 0.3) +
  scale_colour_manual(values = c(Theory = PAL$grey_d),
                                  labels = c(Theory = "Independence theory 1\u2212p"),
                                  name = NULL,
                                  guide = guide_legend(nrow = 1)) +
  ggnewscale::new_scale_colour() +
  geom_errorbar(data = cells,
                aes(xoff, rate, ymin = lo, ymax = hi, colour = bin),
                linewidth = 0.3, width = 0) +
  geom_point(data = cells, aes(xoff, rate, colour = bin), size = 2.2) +
  scale_colour_manual(
    values = c(B1 = PAL$blue_1, B2 = PAL$blue_2, B3 = PAL$blue_sec, B4 = PAL$blue_main),
    labels = c("B1", "B2", "B3", "B4"), name = NULL,
    guide = guide_legend(nrow = 2, byrow = TRUE)) +
  scale_x_continuous(breaks = 1:6, labels = PLEV, limits = c(0.4, 6.8),
                     expand = c(0, 0)) +
  scale_y_log10(limits = c(0.016, 1.05),
                breaks = c(0.02, 0.05, 0.1, 0.2, 0.5),
                labels = c("0.02", "0.05", "0.1", "0.2", "0.5")) +
  labs(x = "Integer accuracy p", y = "Damaged-unit survival") +
  theme_big() +
  theme(legend.position = c(0.02, 0.02), legend.justification = c(0, 0),
        legend.text = element_text(size = 8, colour = "black"),
        legend.key.height = unit(7, "pt"), legend.spacing.y = unit(2, "pt"),
        legend.background = element_blank(),
        legend.box.background = element_blank())
pw <- D$damage_cells$pooled_whole
# v3: inset boxes drawn in p_b's own data space (log-y) so that patchwork can
# align the two panels; the former cowplot::ggdraw wrapper made (b) an opaque
# inset and broke (a)/(b) alignment.
p_b <- p_b +
  annotate("rect", xmin = 3.55, xmax = 4.45, ymin = 0.30, ymax = 0.62,
           fill = PAL$blue_main) +
  annotate("rect", xmin = 5.65, xmax = 6.55, ymin = 0.30, ymax = 0.62,
           fill = "#C9C9C9") +
  annotate("text", x = 4.0, y = 0.41, label = sprintf("%.3f", pw$measured),
           size = 2.6, family = "Times New Roman", colour = "white") +
  annotate("text", x = 6.1, y = 0.41, label = sprintf("%.3f", pw$theory),
           size = 2.6, family = "Times New Roman", colour = "#262626") +
  annotate("text", x = 4.55, y = 0.86, family = "Times New Roman",
           label = "Pooled whole proofs (3,570 draws)", size = 2.6,
           fontface = "bold", hjust = 0.5, colour = PAL$ink)

# v4 (2026-10-03): per user review -- each INDIVIDUAL panel is 1:1 (square
# plot area), panels side by side; canvas returns to a wide strip whose
# height is tuned so both plot areas measure square.
fig <- (p_a | p_b) + plot_annotation(tag_levels = "a")
fig <- fig & theme(plot.tag = element_text(size = 12, face = "bold",
                                           family = "Times New Roman"))
save_pub_r(fig, "output/fig3_law1", width_mm = 162, height_mm = 81,
           panel_ids = c("a", "b"))
