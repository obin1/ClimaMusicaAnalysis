#!/usr/bin/env julia
#
# Recreate the CheMPAS-MUSICA "ABBA" supercell figures (David Fillmore) from a
# ClimaAtmos aquaplanet run.  The 2D geometry differs (sphere lon/lat instead of
# an 84 km periodic box), but the chemistry, the density-dependent equilibrium,
# and the panel set are the same:
#
#   1. qAB / qA time-evolution      -> horizontal slices at z ~ 5 km, 6 times
#   2. qAB / qA vertical evolution  -> lon-z cross-sections at fixed lat, 6 times
#   3. w / u / qAB / qA cross-section at the final time
#   4. Single-cell time series  (z ~ surface / 10 km / 20 km) + box-model overlay
#   5. Domain-average time series                              + box-model overlay
#   6. Mass conservation: mass-weighted volume mean of qAB + qA + qB
#
# Usage:
#   julia --project=.buildkite aquaplanet_ABBA.jl [output_dir]
#
using CairoMakie
using NCDatasets
using Statistics
import Musica

OUTPUT_DIR = get(
    ARGS,
    1,
    "../ClimaAtmos.jl/output/prognostic_edmfx_aquaplanet_abba/output_0021",
)
FIG_DIR = "/Users/psturm/Desktop/ClimaMusicaAnalysis/figs"
mkpath(FIG_DIR)

# ── Chemistry parameters (must match config/chemistry_configs/abba.yaml and the
#    ClimaAtmosMusica extension) ───────────────────────────────────────────────
# The reaction rates/species live in the MICM config below; the box model drives
# the *same* MICM solver, so we don't re-define rate constants here.
CHEM_CONFIG = joinpath(OUTPUT_DIR, "..", "..", "..",
                       "config", "chemistry_configs", "abba.yaml")
isfile(CHEM_CONFIG) ||
    (CHEM_CONFIG = "../ClimaAtmos.jl/config/chemistry_configs/abba.yaml")
M_A = 0.029   # kg/mol
M_B = 0.029   # kg/mol
M_AB = 0.058  # kg/mol
Q_AB0 = 0.6   # initial qAB (DecayingProfile: A=0, B=0, AB=0.6)
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
A = load_var("q_gas_A")
B = load_var("q_gas_B")
AB = load_var("q_gas_AB")
lon, lat, z, times = AB.lon, AB.lat, AB.z, AB.time
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

# ── Box model: 0-D ABBA reactor driven by MICM at fixed air density ρ ────────
# This is the *same* solver the model uses (see ext/ClimaAtmosMusica): we step
# MICM forward in CHEM_DT increments and record at the diagnostic sample times.
# Concentrations are in mol/m^3, conc_X = ρ * q_X / M_X, mirroring the extension
# (which stores ρ_X = ρ_air * q_X and divides by the molar mass).
function box_model(ρ, sample_times; q_AB0 = Q_AB0, dt = CHEM_DT)
    micm = Musica.MICM(; config_path = CHEM_CONFIG)
    state = Musica.create_state(micm)
    Musica.set_conditions!(state; temperatures = 298.15, pressures = 101325)
    Musica.set_user_defined_rate_parameters!(
        state,
        Dict("USER.forward_AB_to_A_B" => 1.0, "USER.reverse_A_B_to_AB" => 1.0),
    )
    Musica.set_concentrations!(
        state,
        Dict("q_gas_A" => 0.0, "q_gas_B" => 0.0, "q_gas_AB" => q_AB0 * ρ / M_AB),
    )

    qAB = similar(sample_times)
    qA = similar(sample_times)
    qB = similar(sample_times)
    t = 0.0
    for (i, ts) in enumerate(sample_times)
        # Independent reference: integrate the pure-chemistry ODE from the true
        # initial condition (qAB = 0.6) to each sample time, with no offset.
        while t < ts - 1e-9
            h = min(dt, ts - t)
            Musica.solve!(micm, state, h)
            t += h
        end
        c = Musica.get_concentrations(state)
        qAB[i] = only(c["q_gas_AB"]) * M_AB / ρ
        qA[i] = only(c["q_gas_A"]) * M_A / ρ
        qB[i] = only(c["q_gas_B"]) * M_B / ρ
    end
    return (; qAB, qA, qB)
end

# ── Helpers ──────────────────────────────────────────────────────────────────
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
# 1. Time-evolution horizontal slices (qAB and qA at z ~ 5 km)
# =============================================================================
function plot_horizontal_evolution(var, name, fname)
    q = var.q
    crange = (minimum(q[:, :, SLICE_Z, :]), maximum(q[:, :, SLICE_Z, :]))
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
    crange = (minimum(q[:, XSEC_LAT, :, :]), maximum(q[:, XSEC_LAT, :, :]))
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

# Animated vertical cross-section with two adjacent panes (e.g. qAB and qA) that
# cycle through every time snapshot in lockstep. Both panes share ONE colorbar,
# with a single colorrange spanning both variables and all frames (fixed — it
# never rescales between frames).
function animate_vertical(v1, n1, v2, n2, fname; framerate = 8)
    q1, q2 = v1.q, v2.q
    s1, s2 = q1[:, XSEC_LAT, :, :], q2[:, XSEC_LAT, :, :]
    crange = (min(minimum(s1), minimum(s2)), max(maximum(s1), maximum(s2)))
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
# 3. w / u / qAB / qA cross-section at the final time
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
    push!(panels, ("qAB [kg/kg]", AB.q[:, XSEC_LAT, :, end], :viridis, false))
    push!(panels, ("qA [kg/kg]", A.q[:, XSEC_LAT, :, end], :viridis, false))

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
    save(joinpath(FIG_DIR, "ABBA_crosssection.png"), fig; px_per_unit = 2)
    println("  saved ABBA_crosssection.png")
end

# =============================================================================
# 4 & 5. Time series with box-model overlay (single cell and domain average)
# =============================================================================
function plot_timeseries(get_series, ttl, fname)
    fig = Figure(size = (1100, 1300), fontsize = 24)
    Label(fig[0, 1], ttl; fontsize = 32, font = :bold)
    # Highest altitude on top, lowest on the bottom (matches the PDF: 20/10/0 km).
    for (row, lev) in enumerate(reverse(LEVELS3))
        qAB_t, qA_t, qB_t = get_series(lev)
        box = box_model(ρ_level[lev], times)
        ax = Axis(fig[row, 1]; xlabel = "Time [min]", ylabel = "mixing ratio [kg/kg]",
                  title = "z ≈ $(round(z[lev] / 1000; digits = 1)) km   (ρ ≈ $(round(ρ_level[lev]; digits = 3)) kg/m³)",
                  titlesize = 28, xlabelsize = 26, ylabelsize = 26,
                  xticklabelsize = 22, yticklabelsize = 22)
        tm = minutes(times)
        # Model output: solid lines (qAB blue, qA teal, qB orange squares).
        lines!(ax, tm, qAB_t; color = :blue, linewidth = 3, label = "qAB")
        lines!(ax, tm, qA_t; color = :teal, linewidth = 3, label = "qA")
        scatter!(ax, tm, qB_t; color = :orange, marker = :rect, markersize = 14, label = "qB")
        lines!(ax, tm, qAB_t .+ qA_t .+ qB_t; color = :black, linewidth = 3, label = "total")
        # Box model (MICM, no transport): dashed.
        lines!(ax, tm, box.qAB; color = :blue, linestyle = :dash, linewidth = 3, label = "qAB (box)")
        lines!(ax, tm, box.qA; color = :teal, linestyle = :dash, linewidth = 3, label = "qA (box)")
        row == 1 && axislegend(ax; position = :rc, nbanks = 2, labelsize = 20)
    end
    save(joinpath(FIG_DIR, fname), fig; px_per_unit = 2)
    println("  saved $fname")
end

# =============================================================================
# 6. Mass conservation: mass-weighted volume mean of qAB + qA + qB
# =============================================================================
function plot_mass_conservation()
    # Weights: area ~ cos(lat), layer thickness Δz, air mass ~ ρ.
    coslat = reshape(cosd.(lat), 1, length(lat), 1)
    zb = vcat(0.0, (z[1:end-1] .+ z[2:end]) ./ 2, z[end] + (z[end] - z[end-1]) / 2)
    Δz = reshape(diff(zb), 1, 1, length(z))
    w = ρ_mean .* coslat .* Δz                       # (lon, lat, z)
    W = sum(w)
    massmean(q) = [sum(w .* q[:, :, :, ti]) / W for ti in 1:nt]
    qABm, qAm, qBm = massmean(AB.q), massmean(A.q), massmean(B.q)

    fig = Figure(size = (900, 500), fontsize = 14)
    ax = Axis(fig[1, 1]; xlabel = "Time [min]", ylabel = "Mass-weighted mean [kg/kg]",
              title = "Mass conservation (volume average, all cells & levels)")
    tm = minutes(times)
    lines!(ax, tm, qABm; color = :blue, label = "qAB")
    lines!(ax, tm, qAm; color = :teal, label = "qA")
    scatter!(ax, tm, qBm; color = :orange, marker = :rect, markersize = 8, label = "qB")
    lines!(ax, tm, qABm .+ qAm .+ qBm; color = :black, linewidth = 2, label = "total")
    ylims!(ax, 0, max(Q_AB0 * 1.1, maximum(qABm .+ qAm .+ qBm) * 1.05))
    axislegend(ax; position = :rc)
    save(joinpath(FIG_DIR, "ABBA_mass_conservation.png"), fig; px_per_unit = 2)
    println("  saved ABBA_mass_conservation.png")
end

# ── Run all panels ───────────────────────────────────────────────────────────
println("Generating figures into $FIG_DIR ...")
plot_horizontal_evolution(AB, "qAB", "ABBA_qAB_evolution.png")
plot_horizontal_evolution(A, "qA", "ABBA_qA_evolution.png")
plot_vertical_evolution(AB, "qAB", "ABBA_qAB_vertical.png")
plot_vertical_evolution(A, "qA", "ABBA_qA_vertical.png")
animate_vertical(AB, "qAB", A, "qA", "ABBA_qAB_qA_vertical.gif")
plot_final_crosssection()

# Single-cell time series at Boulder, Colorado (40.015° N, 105.270° W).
boulder_lon, boulder_lat = -105.270, 40.015
single_lon = nearest(lon, boulder_lon)
single_lat = nearest(lat, boulder_lat)
plot_timeseries(
    lev -> (AB.q[single_lon, single_lat, lev, :],
            A.q[single_lon, single_lat, lev, :],
            B.q[single_lon, single_lat, lev, :]),
    "Single-cell time series — Boulder, CO\n(nearest cell: lon ≈ $(round(lon[single_lon]; digits = 1))°, lat ≈ $(round(lat[single_lat]; digits = 1))°)",
    "ABBA_singlecell.png",
)

# Domain-average time series (simple horizontal mean per level).
domavg(q, lev) = vec(mean(q[:, :, lev, :]; dims = (1, 2)))
plot_timeseries(
    lev -> (domavg(AB.q, lev), domavg(A.q, lev), domavg(B.q, lev)),
    "Domain-average time series (all $(length(lon) * length(lat)) cells)",
    "ABBA_domain_average.png",
)

plot_mass_conservation()
println("Done.")