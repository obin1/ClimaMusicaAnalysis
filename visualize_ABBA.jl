import ClimaAnalysis
import ClimaAnalysis: SimDir, slice
import ClimaAnalysis: Visualize as viz
import CairoMakie

BASE_DIR = "/Users/psturm/Desktop/ClimaAtmos.jl/output/prognostic_edmfx_bomex_ABBA_column"
FIG_DIR  = "/Users/psturm/Desktop/ClimaMusicaAnalysis/figs"

GRID_MEAN = ["q_gas_A", "q_gas_B", "q_gas_AB"]
UPDRAFT   = ["q_gas_Aup", "q_gas_Bup", "q_gas_ABup"]

function load_var(output_tag, short_name; period = "10m")
    simdir = SimDir(joinpath(BASE_DIR, output_tag))
    return get(simdir; short_name, reduction = "inst", period)
end

# Compute a single shared (min, max) colorrange across all output tags and all
# short_names in the group, so tracers within a group share one colorbar scale.
function shared_colorrange(output_tags, short_names; period = "10m")
    lo, hi = Inf, -Inf
    for short_name in short_names, tag in output_tags
        data = load_var(tag, short_name; period).data
        lo = min(lo, minimum(data))
        hi = max(hi, maximum(data))
    end
    return (lo, hi)
end

function plot_ABBA(output_tag, grid_mean_range, updraft_range; period = "10m")
    fig = CairoMakie.Figure(size = (400 * length(GRID_MEAN), 600))

    for (i, short_name) in enumerate(GRID_MEAN)
        var = load_var(output_tag, short_name; period)
        gl = fig[1, i] = CairoMakie.GridLayout()
        more_kwargs = Dict(:plot => Dict(:colorrange => grid_mean_range))
        viz.plot!(gl, var; more_kwargs)
    end

    for (i, short_name) in enumerate(UPDRAFT)
        var = load_var(output_tag, short_name; period)
        gl = fig[2, i] = CairoMakie.GridLayout()
        more_kwargs = Dict(:plot => Dict(:colorrange => updraft_range))
        viz.plot!(gl, var; more_kwargs)
    end

    outpath = joinpath(FIG_DIR, "ABBA_curtains_$(output_tag).png")
    CairoMakie.save(outpath, fig)
    @info "Saved $outpath"
    return fig
end

OUTPUT_TAGS = ["output_0005"]
GRID_MEAN_RANGE = shared_colorrange(OUTPUT_TAGS, GRID_MEAN)
UPDRAFT_RANGE   = shared_colorrange(OUTPUT_TAGS, UPDRAFT)

for tag in OUTPUT_TAGS
    plot_ABBA(tag, GRID_MEAN_RANGE, UPDRAFT_RANGE)
end
