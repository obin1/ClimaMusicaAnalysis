#!/usr/bin/env julia
#
# EDDE analog of aquaplanet_ABBA.jl.  The EDDE mechanism (ED <-> E + D) is
# structurally identical to ABBA (AB <-> A + B); this script exists to verify
# that a mechanism-agnostic ClimaAtmos run named with different species produces
# the same figures.  Everything below mirrors aquaplanet_ABBA.jl with the
# renaming A->E, B->D, AB->ED.
#
# Usage:
#   julia --project=.buildkite aquaplanet_EDDE.jl [output_dir]
#
using CairoMakie
using NCDatasets
using Statistics
import Musica

OUTPUT_DIR = get(
    ARGS,
    1,
    "../ClimaAtmos.jl/output/prognostic_edmfx_aquaplanet_edde/output_0004",
)
FIG_DIR = "/Users/psturm/Desktop/ClimaMusicaAnalysis/figs_edde"
mkpath(FIG_DIR)

# ── Chemistry parameters (must match config/chemistry_configs/edde.yaml and the
#    ClimaAtmosMusica extension) ───────────────────────────────────────────────
# The reaction rates/species live in the MICM config below; the box model drives
# the same MICM solver, so we don't re-define rate constants here.
CHEM_CONFIG = joinpath(OUTPUT_DIR, "..", "..", "..",
                       "config", "chemistry_configs", "edde.yaml")
isfile(CHEM_CONFIG) ||
    (CHEM_CONFIG = "../ClimaAtmos.jl/config/chemistry_configs/edde.yaml")
M_E = 0.029   # kg/mol
M_D = 0.029   # kg/mol
M_ED = 0.058  # kg/mol
Q_ED0 = 0.6   # initial qED (DecayingProfile: E=0, D=0, ED=0.6)
CHEM_DT = 20.0  # chemistry step [s]; match the model dt for an honest overlay
T_MAX = 30 * 60.0  # only plot the first 60 min [s]

# ── Data loading ─────────────────────────────────────────────────────────────
# Instantaneous diagnostics are written as `<name>_<period>_inst.nc` (the period
# depends on the config, e.g. 1m / 2m / 30s); some quantities (e.g. rhoa) may
# only exist as a time average `<name>_<period>_average.nc`. Always prefer an
# instantaneous file (so time-evolution panels get every snapshot), falling back
# to an average only if no inst file exists.
function find_file(name)
    files = readdir(OUTPUT_DIR)
    inst = filter(f -> occursin(Regex("^$(name)_.*_inst\\.nc\$"), f), files)
    !isempty(inst) && return joinpath(OUTPUT_DIR, first(inst))
    avg = filter(f -> occursin(Regex("^$(name)_.*_average\\.nc\$"), f), files)
    !isempty(avg) && return joinpath(OUTPUT_DIR, first(avg))
    return nothing
end

"""Load a remapped variable as (lon, lat, z, time), returning coords too."""
function load_var(short_name; varname = short_name)
    f = find_file(short_name)
    isnothing(f) && error("Could not find a NetCDF file for `$short_name` in $OUTPUT_DIR")
    ds = NCDataset(f)
    lon = Array(ds["lon"][:])
    lat = Array(ds["lat"][:])
    z = Array(ds["z"][:])
    time = Array(ds["time"][:])
    # stored as (time, lon, lat, z) -> permute to (lon, lat, z, time)
    q = permutedims(Array(ds[varname][:, :, :, :]), (2, 3, 4, 1))
    close(ds)
    # Keep only the first T_MAX seconds (e.g. drop the second half of a 1 h run).
    keep = findall(<=(T_MAX + 1e-6), time)
    isempty(keep) && (keep = [1])  # e.g. a single-snapshot hourly average
    time = time[keep]
    q = q[:, :, :, keep]
    return (; lon, lat, z, time, q)
end

println("Loading tracers from $OUTPUT_DIR ...")
E = load_var("q_gas_E")
D = load_var("q_gas_D")
ED = load_var("q_gas_ED")
lon, lat, z, times = ED.lon, ED.lat, ED.z, ED.time
nt = length(times)
println("  grid: $(length(lon)) lon x $(length(lat)) lat x $(length(z)) z, $nt times")

# Air density: prefer instantaneous, else hourly average (broadcast over time).
ρ_field = try
    load_var("rhoa").q
catch
    @warn "rhoa not found; using US-standard-ish ρ(z) = 1.2 * exp(-z/8000)"
    ρ0 = reshape(1.2 .* exp.(-z ./ 8000), 1, 1, length(z), 1)
    repeat(ρ0, length(lon), length(lat), 1, nt)
end
# Time-mean ρ per cell (for the box model / weighting); ρ varies little in time.
ρ_mean = dropdims(mean(ρ_field; dims = 4); dims = 4)  # (lon, lat, z)
# Horizontal-mean ρ per level (for the box-model overlays).
ρ_level = vec(mean(ρ_mean; dims = (1, 2)))            # (z,)

# ── Box model: 0-D EDDE reactor driven by MICM at fixed air density ρ ────────
# This is the *same* solver the model uses (see ext/ClimaAtmosMusica): we step
# MICM forward in CHEM_DT increments and record at the diagnostic sample times.
# Concentrations are in mol/m^3, conc_X = ρ * q_X / M_X, mirroring the extension
# (which stores ρ_X = ρ_air * q_X and divides by the molar mass).
function box_model(ρ, sample_times; q_ED0 = Q_ED0, dt = CHEM_DT)
    micm = Musica.MICM(; config_path = CHEM_CONFIG)
    state = Musica.create_state(micm)
    Musica.set_conditions!(state; temperatures = 298.15, pressures = 101325)
    Musica.set_user_defined_rate_parameters!(
        state,
        Dict("USER.forward_ED_to_E_D" => 1.0, "USER.reverse_E_D_to_ED" => 1.0),
    )
    Musica.set_concentrations!(
        state,
        Dict("q_gas_E" => 0.0, "q_gas_D" => 0.0, "q_gas_ED" => q_ED0 * ρ / M_ED),
    )

    qED = similar(sample_times)
    qE = similar(sample_times)
    qD = similar(sample_times)
    t = 0.0
    for (i, ts) in enumerate(sample_times)
        # Independent reference: integrate the pure-chemistry ODE from the true
        # initial condition (qED = 0.6) to each sample time, with no offset.
        while t < ts - 1e-9
            h = min(dt, ts - t)
            Musica.solve!(micm, state, h)
            t += h
        end
        c = Musica.get_concentrations(state)
        qED[i] = only(c["q_gas_ED"]) * M_ED / ρ
        qE[i] = only(c["q_gas_E"]) * M_E / ρ
        qD[i] = only(c["q_gas_D"]) * M_D / ρ
    end
    return (; qED, qE, qD)
end

# ── Helpers ──────────────────────────────────────────────────────────────────
# Robust color range: ignore non-finite values and guarantee a nonzero width so
# Makie's colormap sampler doesn't divide by zero on a constant field (e.g. an
# all-zero tracer that a run never produced).
function crange_of(data...)
    vals = Float64[]
    for d in data
        append!(vals, filter(isfinite, vec(d)))
    end
    isempty(vals) && return (0.0, 1.0)
    lo, hi = extrema(vals)
    lo == hi && return (lo, lo + (iszero(lo) ? 1.0 : abs(lo) * 1e-6))
    return (lo, hi)
end

nearest(vals, target) = argmin(abs.(vals .- target))
"Pick `n` (<= nt) evenly spaced time indices including first and last."
function snapshot_indices(n)
    n = min(n, nt)
    n == 1 && return [nt]
    return unique(round.(Int, range(1, nt, length = n)))
end
minutes(t) = t ./ 60

TARGET_Z = (0.0, 10_000.0, 20_000.0)        # surface / 10 km / 20 km
LEVELS3 = [nearest(z, zt) for zt in TARGET_Z]
SLICE_Z = nearest(z, 5_000.0)               # ~5 km for horizontal slices
XSEC_LAT = nearest(lat, 0.0)                # equatorial cross-section
SNAP6 = snapshot_indices(6)

# =============================================================================
# 1. Time-evolution horizontal slices (qED and qE at z ~ 5 km)
# =============================================================================
function plot_horizontal_evolution(var, name, fname)
    q = var.q
    crange = crange_of(q[:, :, SLICE_Z, :])
    fig = Figure(size = (1200, 700), fontsize = 14)
    Label(fig[0, 1:3], "$name  —  z ≈ $(round(z[SLICE_Z] / 1000; digits = 1)) km"; fontsize = 18, font = :bold)
    hm = nothing
    for (k, ti) in enumerate(SNAP6)
        r, c = fldmod1(k, 3)
        ax = Axis(fig[r, c]; title = "t = $(round(minutes(times[ti]); digits = 1)) min",
                  xlabel = "Longitude [°]", ylabel = "Latitude [°]")
        hm = heatmap!(ax, lon, lat, q[:, :, SLICE_Z, ti]; colormap = :viridis, colorrange = crange)
    end
    Colorbar(fig[1:2, 4], hm; label = "$name [kg/kg]")
    save(joinpath(FIG_DIR, fname), fig; px_per_unit = 2)
    println("  saved $fname")
end

# =============================================================================
# 2. Vertical evolution (lon-z cross-sections at fixed lat)
# =============================================================================
function plot_vertical_evolution(var, name, fname)
    q = var.q
    crange = crange_of(q[:, XSEC_LAT, :, :])
    zkm = z ./ 1000  # height axis in kilometers
    fig = Figure(size = (1200, 700), fontsize = 14)
    Label(fig[0, 1:3], "$name  —  vertical cross-section (lat ≈ $(round(Int, lat[XSEC_LAT]))°)";
          fontsize = 18, font = :bold)
    hm = nothing
    for (k, ti) in enumerate(SNAP6)
        r, c = fldmod1(k, 3)
        ax = Axis(fig[r, c]; title = "t = $(round(minutes(times[ti]); digits = 1)) min",
                  xlabel = "Longitude [°]", ylabel = "Height [km]")
        hm = heatmap!(ax, lon, zkm, q[:, XSEC_LAT, :, ti]; colormap = :viridis, colorrange = crange)
    end
    Colorbar(fig[1:2, 4], hm; label = "$name [kg/kg]")
    save(joinpath(FIG_DIR, fname), fig; px_per_unit = 2)
    println("  saved $fname")
end

# Animated vertical cross-section with two adjacent panes (e.g. qED and qE) that
# cycle through every time snapshot in lockstep. Both panes share ONE colorbar,
# with a single colorrange spanning both variables and all frames (fixed — it
# never rescales between frames).
function animate_vertical(v1, n1, v2, n2, fname; framerate = 8)
    q1, q2 = v1.q, v2.q
    s1, s2 = q1[:, XSEC_LAT, :, :], q2[:, XSEC_LAT, :, :]
    crange = crange_of(s1, s2)
    zkm = z ./ 1000  # height axis in kilometers
    fig = Figure(size = (1500, 760), fontsize = 28)
    frame = Observable(1)
    suptitle = @lift(
        "Vertical cross-section (lat ≈ $(round(Int, lat[XSEC_LAT]))°)   " *
        "t = $(round(minutes(times[$frame]); digits = 1)) min"
    )
    Label(fig[0, 1:2], suptitle; fontsize = 36, font = :bold)
    hm = nothing
    for (col, (q, n)) in enumerate(((q1, n1), (q2, n2)))
        data = @lift(q[:, XSEC_LAT, :, $frame])
        ax = Axis(fig[1, col]; title = n, xlabel = "Longitude [°]", ylabel = "Height [km]",
                  titlesize = 32, xlabelsize = 28, ylabelsize = 28,
                  xticklabelsize = 24, yticklabelsize = 24)
        hm = heatmap!(ax, lon, zkm, data; colormap = :viridis, colorrange = crange)
    end
    Colorbar(fig[1, 3], hm; label = "mixing ratio [kg/kg]",
             labelsize = 28, ticklabelsize = 24)
    record(fig, joinpath(FIG_DIR, fname), 1:nt; framerate) do i
        frame[] = i
    end
    println("  saved $fname")
end

# =============================================================================
# 3. w / u / qED / qE cross-section at the final time
# =============================================================================
function plot_final_crosssection()
    panels = Any[]
    wfile = find_file("wa")
    ufile = find_file("ua")
    if !isnothing(wfile)
        w = load_var("wa")
        push!(panels, ("Vertical velocity w [m/s]", w.q[:, XSEC_LAT, :, end], :balance, true))
    end
    if !isnothing(ufile)
        u = load_var("ua")
        push!(panels, ("Zonal wind u [m/s]", u.q[:, XSEC_LAT, :, end], :balance, true))
    end
    push!(panels, ("qED [kg/kg]", ED.q[:, XSEC_LAT, :, end], :viridis, false))
    push!(panels, ("qE [kg/kg]", E.q[:, XSEC_LAT, :, end], :viridis, false))

    zkm = z ./ 1000  # height axis in kilometers
    fig = Figure(size = (600 * length(panels), 500), fontsize = 14)
    Label(fig[0, 1:length(panels)],
          "Vertical cross-section (lat ≈ $(round(Int, lat[XSEC_LAT]))°, t = $(round(minutes(times[end]); digits = 1)) min)";
          fontsize = 18, font = :bold)
    for (i, (ttl, data, cmap, diverging)) in enumerate(panels)
        ax = Axis(fig[1, i]; title = ttl, xlabel = "Longitude [°]", ylabel = "Height [km]")
        kw = (; colormap = cmap)
        if diverging
            m = maximum(abs, data)
            kw = (; colormap = cmap, colorrange = (-m, m))
        end
        hm = heatmap!(ax, lon, zkm, data; kw...)
        Colorbar(fig[2, i], hm; vertical = false)
    end
    save(joinpath(FIG_DIR, "EDDE_crosssection.png"), fig; px_per_unit = 2)
    println("  saved EDDE_crosssection.png")
end

# =============================================================================
# 4 & 5. Time series with box-model overlay (single cell and domain average)
# =============================================================================
function plot_timeseries(get_series, ttl, fname)
    fig = Figure(size = (1100, 1300), fontsize = 24)
    Label(fig[0, 1], ttl; fontsize = 32, font = :bold)
    # Highest altitude on top, lowest on the bottom (matches the PDF: 20/10/0 km).
    for (row, lev) in enumerate(reverse(LEVELS3))
        qED_t, qE_t, qD_t = get_series(lev)
        box = box_model(ρ_level[lev], times)
        ax = Axis(fig[row, 1]; xlabel = "Time [min]", ylabel = "mixing ratio [kg/kg]",
                  title = "z ≈ $(round(z[lev] / 1000; digits = 1)) km   (ρ ≈ $(round(ρ_level[lev]; digits = 3)) kg/m³)",
                  titlesize = 28, xlabelsize = 26, ylabelsize = 26,
                  xticklabelsize = 22, yticklabelsize = 22)
        tm = minutes(times)
        # Model output: solid lines (qED blue, qE teal, qD orange squares).
        lines!(ax, tm, qED_t; color = :blue, linewidth = 3, label = "qED")
        lines!(ax, tm, qE_t; color = :teal, linewidth = 3, label = "qE")
        scatter!(ax, tm, qD_t; color = :orange, marker = :rect, markersize = 14, label = "qD")
        lines!(ax, tm, qED_t .+ qE_t .+ qD_t; color = :black, linewidth = 3, label = "total")
        # Box model (MICM, no transport): dashed.
        lines!(ax, tm, box.qED; color = :blue, linestyle = :dash, linewidth = 3, label = "qED (box)")
        lines!(ax, tm, box.qE; color = :teal, linestyle = :dash, linewidth = 3, label = "qE (box)")
        row == 1 && axislegend(ax; position = :rc, nbanks = 2, labelsize = 20)
    end
    save(joinpath(FIG_DIR, fname), fig; px_per_unit = 2)
    println("  saved $fname")
end

# =============================================================================
# 6. Mass conservation: mass-weighted volume mean of qED + qE + qD
# =============================================================================
function plot_mass_conservation()
    # Weights: area ~ cos(lat), layer thickness Δz, air mass ~ ρ.
    coslat = reshape(cosd.(lat), 1, length(lat), 1)
    zb = vcat(0.0, (z[1:end-1] .+ z[2:end]) ./ 2, z[end] + (z[end] - z[end-1]) / 2)
    Δz = reshape(diff(zb), 1, 1, length(z))
    w = ρ_mean .* coslat .* Δz                       # (lon, lat, z)
    W = sum(w)
    massmean(q) = [sum(w .* q[:, :, :, ti]) / W for ti in 1:nt]
    qEDm, qEm, qDm = massmean(ED.q), massmean(E.q), massmean(D.q)

    fig = Figure(size = (900, 500), fontsize = 14)
    ax = Axis(fig[1, 1]; xlabel = "Time [min]", ylabel = "Mass-weighted mean [kg/kg]",
              title = "Mass conservation (volume average, all cells & levels)")
    tm = minutes(times)
    lines!(ax, tm, qEDm; color = :blue, label = "qED")
    lines!(ax, tm, qEm; color = :teal, label = "qE")
    scatter!(ax, tm, qDm; color = :orange, marker = :rect, markersize = 8, label = "qD")
    lines!(ax, tm, qEDm .+ qEm .+ qDm; color = :black, linewidth = 2, label = "total")
    ylims!(ax, 0, max(Q_ED0 * 1.1, maximum(qEDm .+ qEm .+ qDm) * 1.05))
    axislegend(ax; position = :rc)
    save(joinpath(FIG_DIR, "EDDE_mass_conservation.png"), fig; px_per_unit = 2)
    println("  saved EDDE_mass_conservation.png")
end

# ── Run all panels ───────────────────────────────────────────────────────────
println("Generating figures into $FIG_DIR ...")
plot_horizontal_evolution(ED, "qED", "EDDE_qED_evolution.png")
plot_horizontal_evolution(E, "qE", "EDDE_qE_evolution.png")
plot_vertical_evolution(ED, "qED", "EDDE_qED_vertical.png")
plot_vertical_evolution(E, "qE", "EDDE_qE_vertical.png")
animate_vertical(ED, "qED", E, "qE", "EDDE_qED_qE_vertical.gif")
plot_final_crosssection()

# Single-cell time series at Boulder, Colorado (40.015° N, 105.270° W).
boulder_lon, boulder_lat = -105.270, 40.015
single_lon = nearest(lon, boulder_lon)
single_lat = nearest(lat, boulder_lat)
plot_timeseries(
    lev -> (ED.q[single_lon, single_lat, lev, :],
            E.q[single_lon, single_lat, lev, :],
            D.q[single_lon, single_lat, lev, :]),
    "Single-cell time series — Boulder, CO\n(nearest cell: lon ≈ $(round(lon[single_lon]; digits = 1))°, lat ≈ $(round(lat[single_lat]; digits = 1))°)",
    "EDDE_singlecell.png",
)

# Domain-average time series (simple horizontal mean per level).
domavg(q, lev) = vec(mean(q[:, :, lev, :]; dims = (1, 2)))
plot_timeseries(
    lev -> (domavg(ED.q, lev), domavg(E.q, lev), domavg(D.q, lev)),
    "Domain-average time series (all $(length(lon) * length(lat)) cells)",
    "EDDE_domain_average.png",
)

plot_mass_conservation()
println("Done.")
