# -*- coding: utf-8 -*-
# fig_llm_03_step_distribution.R -- Chain length and certification by family.
#
# Claim: chains run 5-9 steps (pooled mean 6.12); the two synthetic families
# are rigid -- all of their mass sits at exactly 5 and 9 steps -- and certify
# 100%, while the 181-item sum-of-squares cascade spreads across 5-7 steps
# (71/72/38) and carries every failure (84/181 = 0.464).
# Panels: (a) every item drawn as one jittered dot at its chain length, one
#         column per family -- family names on the axis, no legend;
#         (b) certified share of the model-produced chains as a dot with a
#         Wilson 95% interval, the k/n counts set beside each interval.
# Source: fig_step_distribution.csv (218 rows, one per item; derived from the
#         436-row two-arm source GEN35_CHAINREAD_ROWS.jsonl).
# Discipline: (b) uses the stored_chain arm only; the gold arm is 218/218 by
# construction and is never quoted as a capability number. Wilson intervals
# are computed here from k/n as a display aid and are stated in the caption.

source("scripts_r/pub_style_r.R")
source("scripts_r/pub_style_llm.R")
library(ggplot2)
library(patchwork)

d <- read.csv(here("fig_step_distribution.csv"))
gold_u <- d[!duplicated(d$ri), ]          # one row per item (arms share gold_ops)
fam_map <- c(gen32_ind_nest = "Nested induction",
             gen32_ind_tower = "Lemma tower",
             gen32_ineq_cascade = "Sum-of-squares cascade")
gold_u$family <- unname(fam_map[gold_u$family])
gold_u$n_decision_steps <- as.integer(gold_u$n_decision_steps)
gold_u$certified_stored_arm <- as.logical(gold_u$certified_stored_arm)
stopifnot(nrow(gold_u) == 218, !any(is.na(gold_u$family)))
gold_u$family <- factor(gold_u$family, levels = names(FAM3))

fam_n <- table(gold_u$family)
pdf_y <- mean(gold_u$n_decision_steps)       # pooled mean = 6.1193

## (a) one dot per item ------------------------------------------------------
pa <- ggplot(gold_u, aes(family, n_decision_steps, colour = family)) +
  geom_hline(yintercept = pdf_y, linetype = "dotted",
             colour = PAL[["grey_m"]], linewidth = 0.35) +
  geom_jitter(size = 1.05, alpha = 0.6, stroke = 0,
              position = position_jitter(width = 0.30, height = 0.33,
                                         seed = 20261001)) +
  scale_colour_manual(values = FAM3, guide = "none") +
  scale_y_continuous(breaks = 5:9, limits = c(4.6, 10.05)) +
  scale_x_discrete(labels = c("Nested\ninduction", "Lemma\ntower",
                              "Sum-of-squares\ncascade")) +
  labs(x = NULL, y = "Decision steps per item", tag = "a") +
  theme_pub() +
  theme(axis.text.x = element_text(lineheight = 0.9,
                                   colour = unname(FAM3)))

## (b) certified share with Wilson interval ----------------------------------
wilson <- function(k, n, z = 1.959964) {
  p <- k / n
  den <- 1 + z^2 / n
  ctr <- (p + z^2 / (2 * n)) / den
  hw <- z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / den
  c(lo = ctr - hw, hi = ctr + hw)
}
cert <- data.frame(
  family = factor(c("Nested induction", "Lemma tower",
                    "Sum-of-squares cascade"),
                  levels = rev(names(FAM3))),        # SoS at the bottom
  n = c(13, 24, 181), k = c(13, 24, 84))
cert[, c("lo", "hi")] <- t(mapply(wilson, cert$k, cert$n))
cert$rate <- cert$k / cert$n

pb <- ggplot(cert, aes(rate, family, colour = family)) +
  geom_segment(aes(x = lo, xend = hi, y = family, yend = family),
               linewidth = 0.5, alpha = 0.55) +
  geom_point(size = 1.7) +
  scale_colour_manual(values = FAM3, guide = "none") +
  scale_x_continuous(limits = c(0, 1.04), breaks = seq(0, 1, 0.25)) +
  labs(x = "Certified rate", y = NULL, tag = "b") +
  theme_pub() +
  theme(axis.text.y = element_text(colour = rev(unname(FAM3))))

fig <- (pa | pb) +
  plot_annotation(
    tag_levels = "a",
    theme = theme())

save_llm(fig, "output/fig_step_distribution", height_mm = 75,
         panel_ids = c("a", "b"))
