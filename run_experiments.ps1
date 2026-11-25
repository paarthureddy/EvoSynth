# ==============================================================================
# EvoSynth Master Experiment Orchestrator (Daily Staged Execution)
# System Profile: 12th Gen Intel Core i5-12450HX (8 cores / 12 threads) | 16GB RAM
# Daily Tracking: Automatically maintains results under results_forest\<TestN_Date>\
# ==============================================================================
param (
    [string]$Stage = "menu",
    [string]$RunName = ""
)

$ErrorActionPreference = "Continue"
$rootDir = $PSScriptRoot
Set-Location $rootDir

# ------------------------------------------------------------------------------
# Daily Run Folder Resolution (e.g., Test1_19_Nov)
# ------------------------------------------------------------------------------
$dateStr = (Get-Date).ToString("dd_MMM")
if (-not $RunName) {
    $existing = Get-ChildItem -Path (Join-Path $rootDir "results_forest") -Directory -Filter "Test*_${dateStr}" -ErrorAction SilentlyContinue | Sort-Object Name
    if ($existing.Count -gt 0) {
        $RunName = $existing[-1].Name
    } else {
        $RunName = "Test1_${dateStr}"
    }
}

$runDir = Join-Path $rootDir "results_forest\$RunName"
$null = New-Item -ItemType Directory -Path "$runDir\baselines" -Force
$null = New-Item -ItemType Directory -Path "$runDir\evosynth" -Force
$null = New-Item -ItemType Directory -Path "$runDir\ablation\cvgs_300" -Force
$null = New-Item -ItemType Directory -Path "$runDir\ablation\cvgs_500" -Force
$null = New-Item -ItemType Directory -Path "$runDir\summary" -Force

function Show-Header {
    Clear-Host
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host "   EVOSYNTH: MASTER EXPERIMENT RUNNER & BENCHMARK ORCHESTRATOR" -ForegroundColor Cyan
    Write-Host "   Active Run Folder : results_forest\$RunName\" -ForegroundColor Yellow
    Write-Host "   Target Benchmark  : UCI Forest Covertype (581,012 samples, 54 features)" -ForegroundColor Cyan
    Write-Host "   System Profile    : Intel Core i5-12450HX (8 Cores/12 Threads) | 16 GB RAM" -ForegroundColor DarkCyan
    Write-Host "   Fault Protection  : Atomic Checkpointing Every 10 Evals with Auto-Resume" -ForegroundColor DarkCyan
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Run-Stage1 {
    Write-Host "`n>>> [STAGE 1] RUNNING BASELINE EXPERIMENTS INTO results_forest\$RunName\baselines\..." -ForegroundColor Yellow
    Write-Host "  1. Default ExtraTrees (Untuned Floor on 60k/20k split)" -ForegroundColor White
    Write-Host "  2. Random Search (120 trials)" -ForegroundColor White
    Write-Host "  3. Optuna TPE Sampler (120 trials, SQLite checkpointed)" -ForegroundColor White
    Write-Host "  4. FLAML AutoML (Time-matched to ~10 mins)" -ForegroundColor White
    Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Gray

    # Optional packages check
    python -c "import optuna" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [!] Optuna not found. Installing optuna..." -ForegroundColor Yellow
        pip install optuna | Out-Null
    }
    python -c "import flaml" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [!] FLAML not found. Installing flaml..." -ForegroundColor Yellow
        pip install flaml | Out-Null
    }

    python scripts/run_baselines_forest.py --test all --run_name $RunName
    Write-Host "`n[STAGE 1 COMPLETE] Baseline logs saved to results_forest\$RunName\baselines\" -ForegroundColor Green
}

function Run-Stage2 {
    Write-Host "`n>>> [STAGE 2] RUNNING EVOSYNTH SCALED BUDGET EXPERIMENTS INTO results_forest\$RunName\evosynth\..." -ForegroundColor Yellow
    Write-Host "  1. EvoSynth @ 300 evaluations (Pop=30, Refit 60k/20k)" -ForegroundColor White
    Write-Host "  2. EvoSynth @ 500 evaluations (Pop=30, Refit 60k/20k)" -ForegroundColor White
    Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Gray

    Write-Host "`n[2A] Starting EvoSynth @ 300 evals (Est: ~20-25 min)..." -ForegroundColor Cyan
    python scripts/run_covertype_evosynth.py --budget 300 --pop_size 30 --checkpoint_interval 10 --run_name $RunName

    Write-Host "`n[2B] Starting EvoSynth @ 500 evals (Est: ~35-45 min)..." -ForegroundColor Cyan
    python scripts/run_covertype_evosynth.py --budget 500 --pop_size 30 --checkpoint_interval 10 --run_name $RunName

    Write-Host "`n[STAGE 2 COMPLETE] Scaled EvoSynth results saved to results_forest\$RunName\evosynth\" -ForegroundColor Green
}

function Run-Stage3 {
    Write-Host "`n>>> [STAGE 3] RUNNING COVERTYPE ABLATION STUDY: CVGS-300 INTO results_forest\$RunName\ablation\cvgs_300\..." -ForegroundColor Yellow
    Write-Host "  Setup: Budget = 300 evals | Pop = 30 | Checkpoint @ 75 evals (25% share)" -ForegroundColor White
    Write-Host "  Algorithms: Isolated GA, PSO, DE, CMA-ES vs. Dynamic Meta-Optimizer" -ForegroundColor White
    Write-Host "  Fault Protection: Checkpoints every 10 evaluations (Safe to interrupt & resume)" -ForegroundColor White
    Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Gray

    python scripts/run_ablation_study_forest.py --setup CVGS-300 --pop_size 30 --max_workers 2 --checkpoint_interval 10 --run_name $RunName
    Write-Host "`n[STAGE 3 COMPLETE] CVGS-300 results saved to results_forest\$RunName\ablation\cvgs_300\" -ForegroundColor Green
}

function Run-Stage4 {
    Write-Host "`n>>> [STAGE 4] RUNNING DEEP COVERTYPE ABLATION STUDY: CVGS-500 (OVERNIGHT)..." -ForegroundColor Yellow
    Write-Host "  Setup: Budget = 500 evals | Pop = 30 | Checkpoint @ 125 evals (25% share)" -ForegroundColor White
    Write-Host "  Algorithms: Isolated GA, PSO, DE, CMA-ES vs. Dynamic Meta-Optimizer" -ForegroundColor White
    Write-Host "  Fault Protection: Checkpoints every 10 evaluations (Safe to interrupt & resume)" -ForegroundColor White
    Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Gray

    python scripts/run_ablation_study_forest.py --setup CVGS-500 --pop_size 30 --max_workers 2 --checkpoint_interval 10 --run_name $RunName
    Write-Host "`n[STAGE 4 COMPLETE] CVGS-500 results saved to results_forest\$RunName\ablation\cvgs_500\" -ForegroundColor Green
}

# Main Selection
if ($Stage -eq "1") { Run-Stage1 }
elseif ($Stage -eq "2") { Run-Stage2 }
elseif ($Stage -eq "3") { Run-Stage3 }
elseif ($Stage -eq "4") { Run-Stage4 }
elseif ($Stage -eq "all") {
    $t0 = Get-Date
    Run-Stage1
    Run-Stage2
    Run-Stage3
    Run-Stage4
    $elapsed = (Get-Date) - $t0
    Write-Host "`n================================================================================" -ForegroundColor Cyan
    Write-Host " ALL STAGES COMPLETED IN $($elapsed.TotalHours.ToString('0.1')) HOURS!" -ForegroundColor Green
    Write-Host " Results are fully structured in results_forest\$RunName\" -ForegroundColor Green
    Write-Host "================================================================================" -ForegroundColor Cyan
}
else {
    Show-Header
    Write-Host "Select which stage to run (Saving into results_forest\$RunName\):" -ForegroundColor White
    Write-Host "  [1] Stage 1: Competitor Baselines (Default ExtraTrees, Optuna, FLAML) [~15-20 min]" -ForegroundColor White
    Write-Host "  [2] Stage 2: EvoSynth Scaled Runs (EvoSynth@300 & EvoSynth@500)      [~1 hour]" -ForegroundColor White
    Write-Host "  [3] Stage 3: Covertype Ablation CVGS-300 (All 5 Algorithms)         [Overnight]" -ForegroundColor White
    Write-Host "  [4] Stage 4: Deep Covertype Ablation CVGS-500 (All 5 Algorithms)    [Overnight]" -ForegroundColor White
    Write-Host "  [A] Run All Stages Sequentially (1 -> 2 -> 3 -> 4)                   [Overnight]" -ForegroundColor Yellow
    Write-Host "  [N] Start a New Test Run Folder (e.g. Test2_$dateStr)" -ForegroundColor Cyan
    Write-Host "  [Q] Quit" -ForegroundColor Red
    Write-Host ""
    $choice = Read-Host "Enter option (1/2/3/4/A/N/Q)"

    switch ($choice.ToUpper()) {
        "1" { Run-Stage1 }
        "2" { Run-Stage2 }
        "3" { Run-Stage3 }
        "4" { Run-Stage4 }
        "A" { 
            Run-Stage1
            Run-Stage2
            Run-Stage3
            Run-Stage4
        }
        "N" {
            $existing = Get-ChildItem -Path (Join-Path $rootDir "results_forest") -Directory -Filter "Test*_${dateStr}" -ErrorAction SilentlyContinue | Sort-Object Name
            $nextIdx = ($existing | Measure-Object).Count + 1
            $newRunName = "Test${nextIdx}_${dateStr}"
            Write-Host "Created new test run target: results_forest\$newRunName\" -ForegroundColor Green
            & $PSCommandPath -Stage "menu" -RunName $newRunName
        }
        default { Write-Host "Exiting." -ForegroundColor Gray }
    }
}
