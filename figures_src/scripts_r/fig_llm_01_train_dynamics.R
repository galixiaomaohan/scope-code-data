# -*- coding: utf-8 -*-
# fig_llm_01_train_dynamics.R -- Training dynamics of the final configuration.
#
# Claim: the 6 000-step run is driven by the objective (4.03 -> 0.025 nats,
# gradient norm 98 -> 1.9) while every held-out probe is dead (operator top-1
# saturates at 1.00 by step 50; the slot-head bias never moves), so progress
# must be read from the loss, not from the eval columns.
#
# Panels: (a) hero: log-scale objective with chain/slot components;
#         (b) gradient norm (log, left) vs the two "dead" probes (right).
# Source: fig_train_dynamics.csv (6000 rows, one per step).
# Notes:  text loss is deliberately not plotted (different normalisation,
#         5.4-8.8 nats, would flatten the panels); it lives in the CSV.

source("scripts_r/pub_style_r.R")
source("scripts_r/pub_style_llm.R")
library(ggplot2)
library(patchwork)
library(scales)

d <- read.csv(here("fig_train_dynamics.csv"))

# 51-step centred moving average, computed with filter() so the window is
# truly centred (no lag); edges use partial windows.
ma <- function(x, w = 51) {
  stats::filter(x, rep(1 / w, w), sides = 2)
}
d$loss_ma <- ma(d$loss); d$chain_ma <- ma(d$chain); d$slot_ma <- ma(d$slot)
d$gn_ma <- ma(d$gn)

# eval points (rows with eval_sel_top1 present)
ev <- d[!is.na(d$eval_sel_top1) & d$eval_sel_top1 != "" &
          !is.na(as.numeric(d$eval_sel_top1)), ]
ev$eval_sel_top1 <- as.numeric(ev$eval_sel_top1)
ev$g_pos <- as.numeric(ev$slot_g_pos)

## (a) objective ------------------------------------------------------------
p1 <- ggplot(d, aes(step)) +
  # raw per-step traces (light grey, no smoothing)
  geom_line(aes(y = loss), colour = PAL[["grey_l"]], linewidth = 0.25,
            alpha = 0.8) +
  geom_line(aes(y = chain), colour = PAL[["grey_l"]], linewidth = 0.25,
            alpha = 0.8) +
  geom_line(aes(y = slot), colour = PAL[["grey_l"]], linewidth = 0.25,
            alpha = 0.8) +
  # smoothed heroes
  geom_line(aes(y = loss_ma, colour = "total objective"), linewidth = 0.55) +
  geom_line(aes(y = chain_ma, colour = "chain (operator sel.)"),
            linewidth = 0.5) +
  geom_line(aes(y = slot_ma, colour = "slot (arguments)"), linewidth = 0.5) +
  scale_colour_manual(
    values = c("total objective" = PAL[["ink"]],
               "chain (operator sel.)" = PAL[["blue_main"]],
               "slot (arguments)" = PAL[["amber"]])) +
  scale_y_log10(
    labels = scales::label_scientific()) +
  coord_cartesian(ylim = c(1e-5, 10), xlim = c(0, 6000)) +
  scale_x_continuous(breaks = seq(0, 6000, 1000)) +
  labs(title = "Objective and its two components",
       x = "Training step", y = "Loss (nats, log scale)",
       tag = "a") +
  theme_pub() +
  guides(colour = guide_legend(nrow = 1, byrow = TRUE,
                               override.aes = list(linewidth = 1.0)))

# annotation block in the empty top-left band (raw traces stay below ~0.1 for
# step > 500); no white halo, nothing is covered
ann <- data.frame(x = 600, y = c(0.55, 0.30),
                  lab = c("4.03 \u2192 0.025 nats", "over 6 000 steps"))
p1 <- p1 + geom_text(data = ann, aes(x, y, label = lab), inherit.aes = FALSE,
                     hjust = 0, size = 2.5, family = "Times New Roman",
                     fontface = "bold", colour = PAL[["ink"]])

## (b) gradient norm + dead probes ------------------------------------------
# right axis: fraction 0..1 mapped to log10-norm range for the dual axis
yl <- c(1e-2, 1.2e2)  # left axis range (log10: -2 .. 2.08)
.fwd <- function(f) {  # fraction -> norm scale
  10^(log10(yl[1]) + f * (log10(yl[2]) - log10(yl[1])))
}
.inv <- function(v) {  # norm -> fraction
  (log10(v) - log10(yl[1])) / (log10(yl[2]) - log10(yl[1]))
}
brks_frac <- c(0, 0.2, 0.4, 0.6, 0.8, 1.0)

p2 <- ggplot(d, aes(step)) +
  geom_line(aes(y = .fwd(ifelse(is.na(slot_g_pos), NA, 0.155)),
                colour = "g_pos: slot-head bias (flat)"), linewidth = 0.5) +
  geom_line(aes(y = gn, colour = "unsmoothed per-step values"),
            linewidth = 0.25, alpha = 0.9) +
  geom_line(aes(y = gn_ma, colour = "gradient norm (51-step mean)"),
            linewidth = 0.55) +
  geom_point(data = ev,
             aes(x = step, y = .fwd(eval_sel_top1),
                 colour = "eval operator top-1"),
             shape = 21, fill = "white", stroke = 0.3, size = 0.9) +
  scale_colour_manual(
    values = c("gradient norm (51-step mean)" = PAL[["amber"]],
               "unsmoothed per-step values"   = PAL[["grey_l"]],
               "g_pos: slot-head bias (flat)" = PAL[["teal"]],
               "eval operator top-1"          = PAL[["blue_sec"]]),
    breaks = c("gradient norm (51-step mean)", "unsmoothed per-step values",
               "g_pos: slot-head bias (flat)", "eval operator top-1"),
    guide = guide_legend(nrow = 1, byrow = TRUE,
                          override.aes = list(               # key glyphs only
                            colour    = c(PAL[["amber"]], PAL[["grey_m"]],
                                          PAL[["teal"]], PAL[["blue_sec"]]),
                            linewidth = c(0.55, 0.3, 0.5, 0),
                            shape     = c(NA, NA, NA, 21),
                            fill      = c(NA, NA, NA, "white")))) +
  scale_y_log10(limits = yl, breaks = c(1e-2, 1e-1, 1e0, 1e1, 1e2),
                labels = c("0.01", "0.1", "1", "10", "100"),
                sec.axis = sec_axis(~ .inv(.), name = "Eval accuracy (fraction)",
                                    breaks = brks_frac,
                                    labels = scales::number_format(
                                      accuracy = 0.1))) +
  coord_cartesian(xlim = c(0, 6000)) +
  scale_x_continuous(breaks = seq(0, 6000, 1000)) +
  labs(title = "Gradient norm vs the two dead probes",
       x = "Training step", y = "Gradient norm (log scale)",
       tag = "b") +
  theme_pub()

fig <- p1 + p2 + plot_layout(ncol = 2, widths = c(1, 1), guides = "collect") +
  plot_annotation(
    theme = theme(
      legend.position = "bottom",
      legend.box      = "vertical",
      legend.text     = element_text(size = 6.4, family = "Times New Roman"),
      plot.caption    = element_text(size = 6.2, hjust = 0,
                                     family = "Times New Roman",
                                     colour = PAL[["grey_d"]],
                                     lineheight = 1.08),
      plot.title      = element_text(size = 8.5, face = "bold",
                                     family = "Times New Roman")),
    title = "Final-configuration training dynamics (6 000 steps)",
    caption = paste0(
      "Grey: unsmoothed per-step values; the text loss is not plotted ",
      "(different normalisation and scale, 5.4-8.8 nats).\n",
      "Both held-out probes are dead -- operator top-1 saturates at 1.00 by ",
      "step 50 and g_pos stays flat -- so progress is read from the loss."))

save_llm(fig, "output/fig_train_dynamics", height_mm = 94,
         panel_ids = c("a", "b"))
