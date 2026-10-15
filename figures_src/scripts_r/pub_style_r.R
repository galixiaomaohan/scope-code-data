# -*- coding: utf-8 -*-
# pub_style_r.R -- shared style, palette and export helpers for the manuscript
# figures (R/ggplot2 track).
#
# Premium look: Times New Roman, hairline axes, frameless legends, restrained
# blue/grey family with red/green reserved for contrast semantics.

# Optional geometry auditor from the local figure toolchain. Reproducing the
# figures needs only the manifest writer below; the strict gate runs when the
# auditor is available and is skipped with a message otherwise.
PANEL_AUDIT_SCRIPTS <- Sys.getenv("PANEL_AUDIT_SCRIPTS", unset = NA_character_)

# Patchwork measures label metrics on the auto-opened default pdf device,
# which only knows base-14 faces; alias "Times New Roman" to base-14 Times
# (identical font metrics). UTF-8 glyphs like ≠ are rendered by cairo_pdf /
# svglite / ragg in the real exports; the cairo probe keeps their widths
# faithful for geometry measurement.
if (!"Times New Roman" %in% names(grDevices::pdfFonts())) {
  grDevices::pdfFonts(`Times New Roman` = grDevices::pdfFonts("Times")[[1]])
}

PAL <- list(
  blue_main = "#0F4D92",
  blue_sec  = "#3775BA",
  blue_1    = "#B9D2EA",
  blue_2    = "#7FA8D0",
  grey_l    = "#CFCECE",
  grey_m    = "#8A8A8A",
  grey_d    = "#474747",
  red_s     = "#B64342",
  green_s   = "#3E8E55",
  tol_red   = "#EE6677",
  tol_green = "#228833",
  ink       = "#262626"
)

theme_pub <- function(base_size = 8) {
  theme_classic(base_size = base_size, base_family = "Times New Roman") +
    theme(
      axis.line        = element_line(linewidth = 0.4, colour = "black"),
      axis.ticks       = element_line(linewidth = 0.4, colour = "black"),
      axis.ticks.length = unit(2, "pt"),
      axis.title       = element_text(size = base_size, colour = "black"),
      axis.text        = element_text(size = base_size - 0.5, colour = "black"),
      legend.title     = element_blank(),
      legend.text      = element_text(size = base_size - 0.5, colour = "black"),
      legend.key       = element_blank(),
      legend.key.width = unit(10, "pt"),
      plot.tag         = element_text(size = 9.5, face = "bold",
                                      family = "Times New Roman"),
      plot.tag.position = "topleft"
    )
}

# Export helper: patchwork alignment gate (multi-panel) + svg/pdf/png bundle.
#
# NOTE on the manifest measurement: with ggplot2 4.x, patchworkGrob() keeps the
# panel areas as *null* units, which grid::convertWidth() resolves to 0 outside
# the gtable's own layout context (the stock helper therefore measures
# zero-size panels). write_patchwork_panel_layout_manifest() below keeps the
# reference toolchain's manifest schema, panel-name pattern and auditor invocation identical,
# but resolves null units explicitly the way gtable lays them out at draw time
# (leftover physical space shared proportionally by null magnitude).

.nature_resolve_units <- function(u, total_pt, is_width) {
  types <- grid::unitType(u)
  vals <- as.numeric(u)
  conv <- if (is_width) grid::convertWidth else grid::convertHeight
  abs_pt <- rep(0, length(u))
  abs_idx <- types != "null"
  if (any(abs_idx)) {
    abs_pt[abs_idx] <- conv(u[abs_idx], "pt", valueOnly = TRUE)
  }
  leftover <- total_pt - sum(abs_pt)
  null_sum <- sum(vals[!abs_idx])
  if (null_sum > 0 && leftover > 0) {
    abs_pt[!abs_idx] <- leftover * vals[!abs_idx] / null_sum
  }
  abs_pt
}

write_patchwork_panel_layout_manifest <- function(
  plot, manifest_path, width_in, height_in, panel_ids = NULL,
  row_groups = NULL, column_groups = NULL, exemptions = list()) {
  for (package in c("patchwork", "grid", "jsonlite")) {
    if (!requireNamespace(package, quietly = TRUE)) {
      stop(sprintf("Package '%s' is required for patchwork alignment QA", package), call. = FALSE)
    }
  }
  grob <- patchwork::patchworkGrob(plot)
  rows <- grob$layout[grepl("^panel(?:-[0-9]+)?$", grob$layout$name, perl = TRUE), , drop = FALSE]
  rows <- rows[order(rows$t, rows$l), , drop = FALSE]
  if (nrow(rows) < 2) {
    stop("At least two patchwork panel cells are required for alignment QA", call. = FALSE)
  }
  if (is.null(panel_ids)) {
    panel_ids <- vapply(seq_len(nrow(rows)), function(i)
      if (i <= length(letters)) letters[i] else paste0("panel-", i), character(1))
  }
  panel_ids <- as.character(panel_ids)
  if (length(panel_ids) != nrow(rows) || anyDuplicated(panel_ids) || any(!nzchar(panel_ids))) {
    stop("panel_ids must be unique, non-empty, and match the measured patchwork panels", call. = FALSE)
  }

  probe_path <- tempfile(fileext = ".pdf")
  grDevices::cairo_pdf(probe_path, width = width_in, height = height_in)
  grid::grid.newpage()
  grid::grid.draw(grob)
  grid::grid.force()
  widths_pt <- .nature_resolve_units(grob$widths, width_in * 72, is_width = TRUE)
  heights_pt <- .nature_resolve_units(grob$heights, height_in * 72, is_width = FALSE)
  grDevices::dev.off()
  unlink(probe_path)
  if (any(!is.finite(widths_pt)) || any(!is.finite(heights_pt))) {
    stop("Patchwork panel dimensions could not be converted to physical points", call. = FALSE)
  }

  x_edges <- c(0, cumsum(widths_pt))
  top_edges <- c(0, cumsum(heights_pt))
  total_height_pt <- sum(heights_pt)
  panels <- lapply(seq_len(nrow(rows)), function(index) {
    row <- rows[index, ]
    left <- x_edges[row$l]; right <- x_edges[row$r + 1]
    top <- total_height_pt - top_edges[row$t]
    bottom <- total_height_pt - top_edges[row$b + 1]
    list(id = panel_ids[index],
         bbox_pt = unname(c(left, bottom, right, top)),
         grid_id = "patchwork-grid-1",
         row_start = as.integer(row$t - 1), row_stop = as.integer(row$b),
         col_start = as.integer(row$l - 1), col_stop = as.integer(row$r))
  })

  manifest <- list(
    schema_version = 1L, backend = "r-patchwork",
    figure = list(width_pt = width_in * 72, height_pt = height_in * 72),
    panels = panels, exemptions = exemptions)
  if (!is.null(row_groups)) {
    manifest$row_groups <- lapply(row_groups, function(g) as.character(unlist(g, use.names = FALSE)))
  }
  if (!is.null(column_groups)) {
    manifest$column_groups <- lapply(column_groups, function(g) as.character(unlist(g, use.names = FALSE)))
  }
  dir.create(dirname(manifest_path), recursive = TRUE, showWarnings = FALSE)
  jsonlite::write_json(manifest, manifest_path, auto_unbox = TRUE, pretty = TRUE, digits = NA)
  invisible(manifest)
}

save_pub_r <- function(plot, filename, width_mm, height_mm, dpi = 600,
                       panel_ids = NULL, row_groups = NULL, column_groups = NULL,
                       exemptions = list()) {
  dir.create(dirname(filename), recursive = TRUE, showWarnings = FALSE)
  w <- width_mm / 25.4
  h <- height_mm / 25.4
  if (inherits(plot, "patchwork")) {
    write_patchwork_panel_layout_manifest(
      plot,
      manifest_path = paste0(filename, ".alignment-layout.json"),
      width_in  = w,
      height_in = h,
      panel_ids      = panel_ids,
      row_groups     = row_groups,
      column_groups  = column_groups,
      exemptions     = exemptions)

    audit_script <- if (is.na(PANEL_AUDIT_SCRIPTS)) NULL
      else file.path(PANEL_AUDIT_SCRIPTS, "audit_panel_alignment.py")
    if (is.null(audit_script) || !file.exists(audit_script)) {
      message("alignment: manifest written; strict audit skipped (auditor not installed)")
    } else {
      report_path <- paste0(filename, ".alignment.json")
      overlay_svg <- paste0(filename, ".alignment.svg")
      args <- c(shQuote(audit_script), shQuote(paste0(filename, ".alignment-layout.json")),
                "--json-out", shQuote(report_path),
                "--tolerance-pt", "1.5", "--gutter-tolerance-pt", "1.5",
                "--overlay-svg", shQuote(overlay_svg), "--strict")
      output <- suppressWarnings(system2(Sys.which("python"), args = args,
                                         stdout = TRUE, stderr = TRUE))
      status <- attr(output, "status")
      if (is.null(status)) status <- 0L
      if (length(output)) message(paste(output, collapse = "\n"))
      if (status != 0L) {
        stop(sprintf("Panel alignment gate failed with exit code %d; inspect %s",
                     status, report_path), call. = FALSE)
      }
    }
  } else {
    message("alignment: NOT APPLICABLE (single plot area)")
  }
  svglite::svglite(paste0(filename, ".svg"), width = w, height = h)
  print(plot); dev.off()
  grDevices::cairo_pdf(paste0(filename, ".pdf"), width = w, height = h,
                       family = "Times New Roman")
  print(plot); dev.off()
  ragg::agg_png(paste0(filename, ".png"), width = w, height = h,
                units = "in", res = dpi)
  print(plot); dev.off()
  # Audit-only PDF via the base-14 pdf() device: cairo_pdf encodes glyph size
  # through the text matrix with `1 Tf`, which the geometry auditors cannot
  # measure; pdf() writes real Tf sizes with identical Times metrics. Non-WinAnsi
  # glyphs (≈, ≠) substitute there, so this copy is QA-only, never delivered.
  grDevices::pdf(paste0(filename, ".audit.pdf"), width = w, height = h,
                 family = "Times New Roman", useDingbats = FALSE)
  print(plot); dev.off()
  invisible(TRUE)
}

# Convention: every figure script is run with working directory
# figures_src/ (e.g.  Rscript scripts_r/fig_llm_06_hidden_trajectory.R),
# so outputs land in figures_src/ and data paths resolve as
# "data_cited.json", "../data/processed/figures/...", and "../data/raw/...".
