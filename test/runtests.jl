using GREB
using Test

# Smoke tests that do NOT require the (large, external) JDAL2 input data.
# They check that the package loads, its types build, and the grid/constants
# are intact after the notebook -> package extraction. Full integration runs
# (which need `greb_dataset_jdal2/`) are demonstrated in examples/run_greb.jl.

@testset "GREB.jl" begin

    @testset "grid constants" begin
        @test GREB.xdim == 96
        @test GREB.ydim == 48
        @test GREB.nstep_yr == 730
    end

    @testset "PhysicsConfig" begin
        cfg = PhysicsConfig()
        @test cfg isa PhysicsConfig

        for exp in (:full_model, :constant_topo, :co2_double, :co2_quadruple,
                    :elnino, :lanina, :rcp85)
            c = create_experiment_config(exp)
            @test c isa PhysicsConfig
            @test c.experiment == exp
        end

    end

    @testset "CO₂ experiments keep the control CO₂ unperturbed" begin
        # Regression: presets used to set co2_concentration (= control CO₂) to the
        # scenario value, so control and scenario ran at the same CO₂.
        for (exp, scnr_co2) in ((:co2_double, 680.0), (:co2_quadruple, 1360.0),
                                (:paleo_231kyr, 200.0))
            cfg = create_experiment_config(exp)
            @test cfg.co2_concentration == 340.0
            @test forcing(1, 1950, cfg).CO2 == scnr_co2
            ini = redirect_stderr(devnull) do
                init_model!(cfg)
            end
            @test ini.CO2_ctrl == 340.0
        end
    end

    @testset "hydrology parameters" begin
        # Regression: coefficients used to be written to module globals (never read),
        # and the lookup table was a Tuple indexed by position.
        expected = Dict(-1 => (1.0, 0.0, 0.0, 0.0),
                        0 => (-1.88, 2.25, -17.69, 59.07),
                        1 => (-1.391649, 3.018774, 0.0, 0.0),
                        2 => (0.862162, 0.0, -29.02096, 0.0),
                        3 => (-0.2685845, 1.4591853, -26.9858807, 0.0))
        for (lr, coeffs) in expected
            cfg = PhysicsConfig(log_rain = lr)
            @test_logs (:info,) set_hydrology_parameters!(cfg)
            @test (cfg.c_q, cfg.c_rq, cfg.c_omega, cfg.c_omegastd) == coeffs
        end

        cfg = PhysicsConfig(log_rain = 0, log_clim = 1)  # NCEP coefficients
        @test_logs (:info,) set_hydrology_parameters!(cfg)
        @test (cfg.c_q, cfg.c_rq, cfg.c_omega, cfg.c_omegastd) == (-1.27, 1.99, -16.54, 21.15)

        @test_throws ArgumentError set_hydrology_parameters!(PhysicsConfig(log_rain = 7))

        # Warn when log_clim disagrees with the loaded dataset
        try
            GREB.LOADED_DATASET[] = :ncep
            @test_logs (:warn, r"do not match") (:info,) set_hydrology_parameters!(PhysicsConfig(log_clim = 0))
            @test_logs (:info,) set_hydrology_parameters!(PhysicsConfig(log_clim = 1))
            @test_logs (:info,) set_hydrology_parameters!(PhysicsConfig(log_rain = -1, log_clim = 0))
        finally
            GREB.LOADED_DATASET[] = :none
        end

        # The coefficients actually reach hydro!: rain differs between schemes
        rain(lr) = begin
            cfg = PhysicsConfig(log_rain = lr)
            redirect_stderr(devnull) do; set_hydrology_parameters!(cfg); end
            Ts = fill(290.0, GREB.xdim, GREB.ydim); q = fill(0.008, GREB.xdim, GREB.ydim)
            copy(hydro!(Ts, q, TimeState(1, 1), cfg, CirculationWorkspace()).dq_rain)
        end
        @test rain(-1) != rain(1)
    end

    @testset "workspaces & accumulators" begin
        ws = CirculationWorkspace()
        @test ws isa CirculationWorkspace

        acc = MonthlyAccumulator()
        @test acc isa MonthlyAccumulator
        @test (GREB.reset!(acc); true)   # reset! runs without error

        ts = TimeState(1, 1)
        @test ts.jday == 1
        @test ts.ityr == 1
    end

    @testset "MonthlyRecord type" begin
        @test MonthlyRecord <: NamedTuple
        @test :Ts in fieldnames(MonthlyRecord)
        @test :precip in fieldnames(MonthlyRecord)
    end

    @testset "greb_model! runs without notebook globals" begin
        # Regression: qflux_correction!/greb_model! used to reference the Pluto
        # @bind globals `time_flux`/`jdal2_dir`. They are now parameters, so the
        # model must run to completion on default (unloaded) fields. Values are
        # NaN without real JDAL2 data — we only assert it runs and shapes are OK.
        cfg = create_experiment_config(:full_model)
        result = redirect_stdout(devnull) do
            greb_model!(0, 1, 0, cfg; jdal2_dir = "")
        end
        @test length(result.ctrl) == 12
        @test length(result.scnr) == 0
        @test result.ctrl[1] isa MonthlyRecord
    end

    @testset "read_jdal2 rejects non-JDAL2 input" begin
        tmp = tempname()
        write(tmp, "not a jdal2 file")
        @test_throws Exception read_jdal2(tmp)
        rm(tmp; force = true)
    end

end
