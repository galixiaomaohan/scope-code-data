# -*- coding: utf-8 -*-
# fig_llm_04_v15_ladder.R -- Depth-curriculum ladder of run dyn_latq_v15.
#
# Claim: the gate-probe depth unlocks monotonically 1 -> 16 over 15 events (all
# after step 11 000) while the probe error falls to 0.167 -- but the sampler
# keeps drawing shallow episodes, so the ladder measures the gate, not the
# training mixture.
# Panels: (a) gate step function + sampled episode depth backdrop + event
#             verticals; (b) probe vs training final-step error, log scale;
#         (c) sampled training-episode depth T over time.
# Source: fig_v15_ladder_gate/train/events.csv parsed from dyn_latq_v15.log.
# Discipline: the log's two "T" quantities stay separate marks; 0.096 is a T=3
# episode reading and is never quoted as the depth-16 error (0.167 is).

source("scripts_r/pub_style_r.R")
source("scripts_r/pub_style_llm.R")
library(ggplot2)
library(patchwork)

gate  <- read.csv(here("fig_v15_ladder_gate.csv"))
train <- read.csv(here("fig_v15_ladder_train.csv"))
ev    <- read.csv(here("fig_v15_ladder_events.csv"))

unlock_steps <- ev$step[ev$event == "DEPTH_UNLOCK"]
stall_steps  <- ev$step[ev$event == "STALL"]
stopifnot(length(unlock_steps) == 15, length(stall_steps) == 3)

train$efin_ma <- as.numeric(stats::filter(train$efin, rep(1 / 21, 21),
                                          sides = 2))
xbr <- seq(0, 15000, 5000)
xlim <- c(0, 19200)

## (a) the ladder -------------------------------------------------------------
pa <- ggplot() +
  geom_point(data = train, aes(step, train_depth_T),
             colour = PAL[["grey_l"]], size = 0.4, alpha = 0.9) +
  geom_vline(xintercept = unlock_steps, linetype = "dotted",
             linewidth = 0.25, colour = PAL[["blue_sec"]], alpha = 0.5) +
  geom_vline(xintercept = stall_steps, linetype = "dotted",
             linewidth = 0.3, colour = PAL[["red_s"]], alpha = 0.75) +
  geom_step(data = gate, aes(step, gate_tmax), colour = PAL[["blue_main"]],
            linewidth = 0.5) +
  geom_point(data = gate, aes(step, gate_tmax), shape = 21, fill = "white",
             colour = PAL[["blue_main"]], size = 0.65, stroke = 0.22) +
  annotate("text", x = 300, y = 15.9, hjust = 0, size = 2.1,
           family = "Times New Roman", colour = PAL[["grey_d"]],
           label = "grey dots: sampled episode depth T") +
  annotate("text", x = 300, y = 14.3, hjust = 0, size = 2.1,
           family = "Times New Roman", fontface = "bold",
           colour = PAL[["blue_main"]], label = "blue: gate probe depth") +
  annotate("text", x = unlock_steps[1] + 220, y = 12.9, angle = 90, hjust = 0,
           size = 1.9, family = "Times New Roman", colour = PAL[["blue_sec"]],
           label = "unlock") +
  annotate("text", x = stall_steps + 220, y = 12.9, angle = 90, hjust = 0,
           size = 1.9, family = "Times New Roman", colour = PAL[["red_s"]],
           label = "stall") +
  scale_x_continuous(limits = xlim, breaks = xbr) +
  scale_y_continuous(breaks = c(2, 4, 6, 8, 10, 12, 14, 16)) +
  labs(title = "Depth curriculum: gate climbs 1 \u2192 16",
       x = "Training step", y = "Depth", tag = "a") +
  theme_pub()

## (b) final-step error -------------------------------------------------------
pb <- ggplot() +
  geom_line(data = train, aes(step, efin_ma), colour = PAL[["amber"]],
            linewidth = 0.5, na.rm = TRUE) +
  geom_line(data = gate, aes(step, e_fin_at_tmax), colour = PAL[["teal"]],
            linewidth = 0.5) +
  geom_point(data = gate, aes(step, e_fin_at_tmax), shape = 21, fill = "white",
             colour = PAL[["teal"]], size = 0.55, stroke = 0.2) +
  annotate("text", x = 18750, y = 0.36, hjust = 1, size = 2.25,
           family = "Times New Roman", fontface = "bold",
           colour = PAL[["teal"]], label = "probe 0.167") +
  annotate("text", x = 2900, y = 0.112, hjust = 0, size = 2.25,
           family = "Times New Roman", fontface = "bold",
           colour = PAL[["amber"]], label = "training e_fin, 21-step mean") +
  scale_x_continuous(limits = xlim, breaks = xbr) +
  scale_y_log10(limits = c(0.06, 2.5), breaks = c(0.1, 0.3, 1),
                labels = c("0.1", "0.3", "1")) +
  labs(title = "Final-step error (log scale)", x = "Training step",
       y = expression(italic(e)[fin]), tag = "b") +
  theme_pub()

## (c) sampled training depth -------------------------------------------------
pc <- ggplot(train, aes(step, train_depth_T)) +
  geom_point(colour = PAL[["amber"]], size = 0.45, alpha = 0.8,
             position = position_jitter(width = 0, height = 0.28, seed = 42)) +
  annotate("text", x = 300, y = 14.8, hjust = 0, size = 2.25,
           family = "Times New Roman", fontface = "bold", colour = PAL[["ink"]],
           label = "241 of 377 episodes sampled at T = 1") +
  scale_x_continuous(limits = xlim, breaks = xbr) +
  # "%3d" pads the tick labels to the same box as (b)'s "0.1"/"0.3" so the
  # measured inter-panel gutters stay within the alignment gate tolerance;
  # the padding is invisible because y tick labels are right-aligned.
  scale_y_continuous(breaks = c(1, 5, 9, 13),
                     labels = function(x) sprintf("%3d", x)) +
  labs(title = "Sampled training-episode depth", x = "Training step",
       y = "Sampled depth T", tag = "c") +
  theme_pub()

fig <- (pa | pb | pc) +
  plot_annotation(
    tag_levels = "a",
    caption = paste0(
      "Two distinct T readings, kept as separate marks: blue = gate-probe ",
      "depth t_max; grey/amber = sampled training-episode depth.\n",
      "The gate reading at depth 16 is 0.167 (probe); 0.096 is a T = 3 ",
      "training-step reading and is not the depth-16 result."),
    theme = theme(
      plot.caption = element_text(size = 6.2, hjust = 0,
                                  family = "Times New Roman",
                                  colour = PAL[["grey_d"]],
                                  lineheight = 1.08)))

save_llm(fig, "output/fig_v15_ladder", height_mm = 86,
         panel_ids = c("a", "b", "c"))
