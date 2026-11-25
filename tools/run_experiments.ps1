# ==============================================================================
# EvoSynth Master Experiment Orchestrator (Staged Execution)
# System Profile: 12th Gen Intel Core i5-12450HX (8 cores / 12 threads) | 16GB RAM
# Every run writes to results\<experiment>\<dataset>\<variant>\seed_<k>\
# (manifest.json, result.json, run.log, ...). Re-running a stage overwrites that
# stage's run folders; the ablation and Optuna stages resume from their checkpoints.
# ==============================================================================
param (
    [string]$Stage = "menu",
    # Extra arguments passed to every evosynth command, e.g. -Extra "--set","run.random_state=7"
    [string[]]$Extra = @()
)

$ErrorActionPreference = "Continue"
# This script lives in tools/; the project root is its parent folder.
$rootDir = Split-Path -Parent $PSScriptRoot
Set-Location $rootDir

# EvoSynth is installed for Python 3.14 (py -3.14 -m pip install -e .).
# Budgets, population sizes, checkpoint intervals, etc. come from settings/*.toml;
# override any of them per run with  --set section.key=value
function Invoke-EvoSynth { & py -3.14 -m evosynth @args @Extra }

function Show-Header {
    Clear-Host
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host "   EVOSYNTH: MASTER EXPERIMENT RUNNER & BENCHMARK ORCHESTRATOR" -ForegroundColor Cyan
    Write-Host "   Results Folder    : results\<experiment>\covertype\<variant>\seed_<k>\" -ForegroundColor Yellow
    Write-Host "   Target Benchmark  : UCI Forest Covertype (581,012 samples, 54 features)" -ForegroundColor Cyan
    Write-Host "   System Profile    : Intel Core i5-12450HX (8 Cores/12 Threads) | 16 GB RAM" -ForegroundColor DarkCyan
    Write-Host "   Fault Protection  : Atomic Checkpointing Every 10 Evals with Auto-Resume" -ForegroundColor DarkCyan
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Run-Stage1 {
    Write-Host "`n>>> [STAGE 1] RUNNING BASELINE EXPERIMENTS INTO results\baselines\covertype\..." -ForegroundColor Yellow
    Write-Host "  1. Default ExtraTrees (Untuned Floor on 60k/20k split)" -ForegroundColor White
    Write-Host "  2. Random Search (120 trials)" -ForegroundColor White
    Write-Host "  3. Optuna TPE Sampler (120 trials, SQLite checkpointed)" -ForegroundColor White
    Write-Host "  4. FLAML AutoML (Time-matched to ~10 mins)" -ForegroundColor White
    Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Gray

    # Optional packages check
    py -3.14 -c "import optuna" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [!] Optuna not found. Installing optuna..." -ForegroundColor Yellow
        py -3.14 -m pip install optuna | Out-Null
    }
    py -3.14 -c "import flaml" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [!] FLAML not found. Installing flaml..." -ForegroundColor Yellow
        py -3.14 -m pip install flaml | Out-Null
    }

    Invoke-EvoSynth baselines --test all
    Write-Host "`n[STAGE 1 COMPLETE] Baseline runs saved under results\baselines\covertype\" -ForegroundColor Green
}

function Run-Stage2 {
    Write-Host "`n>>> [STAGE 2] RUNNING EVOSYNTH SCALED BUDGET EXPERIMENTS INTO results\scaling\covertype\..." -ForegroundColor Yellow
    Write-Host "  1. EvoSynth @ 300 evaluations (Pop=30, Refit 60k/20k)" -ForegroundColor White
    Write-Host "  2. EvoSynth @ 500 evaluations (Pop=30, Refit 60k/20k)" -ForegroundColor White
    Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Gray

    Write-Host "`n[2A] Starting EvoSynth @ 300 evals (Est: ~20-25 min)..." -ForegroundColor Cyan
    Invoke-EvoSynth scale --budget 300

    Write-Host "`n[2B] Starting EvoSynth @ 500 evals (Est: ~35-45 min)..." -ForegroundColor Cyan
    Invoke-EvoSynth scale --budget 500

    Write-Host "`n[STAGE 2 COMPLETE] Scaled EvoSynth runs saved under results\scaling\covertype\" -ForegroundColor Green
}

function Run-Stage3 {
    Write-Host "`n>>> [STAGE 3] RUNNING COVERTYPE ABLATION STUDY: CVGS-300 INTO results\ablation\covertype\cvgs_300\..." -ForegroundColor Yellow
    Write-Host "  Setup: Budget = 300 evals | Pop = 30 | Checkpoint @ 75 evals (25% share)" -ForegroundColor White
    Write-Host "  Algorithms: Isolated GA, PSO, DE, CMA-ES vs. Dynamic Meta-Optimizer" -ForegroundColor White
    Write-Host "  Fault Protection: Checkpoints every 10 evaluations (Safe to interrupt & resume)" -ForegroundColor White
    Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Gray

    Invoke-EvoSynth ablation --setup CVGS-300
    Write-Host "`n[STAGE 3 COMPLETE] CVGS-300 saved under results\ablation\covertype\cvgs_300\" -ForegroundColor Green
}

function Run-Stage4 {
    Write-Host "`n>>> [STAGE 4] RUNNING DEEP COVERTYPE ABLATION STUDY: CVGS-500 (OVERNIGHT)..." -ForegroundColor Yellow
    Write-Host "  Setup: Budget = 500 evals | Pop = 30 | Checkpoint @ 125 evals (25% share)" -ForegroundColor White
    Write-Host "  Algorithms: Isolated GA, PSO, DE, CMA-ES vs. Dynamic Meta-Optimizer" -ForegroundColor White
    Write-Host "  Fault Protection: Checkpoints every 10 evaluations (Safe to interrupt & resume)" -ForegroundColor White
    Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Gray

    Invoke-EvoSynth ablation --setup CVGS-500
    Write-Host "`n[STAGE 4 COMPLETE] CVGS-500 saved under results\ablation\covertype\cvgs_500\" -ForegroundColor Green
}

function Run-All {
    $t0 = Get-Date
    Run-Stage1
    Run-Stage2
    Run-Stage3
    Run-Stage4
    $elapsed = (Get-Date) - $t0
    Write-Host "`n================================================================================" -ForegroundColor Cyan
    Write-Host " ALL STAGES COMPLETED IN $($elapsed.TotalHours.ToString('0.1')) HOURS!" -ForegroundColor Green
    Write-Host " Results are under results\ (one folder per run)" -ForegroundColor Green
    Write-Host "================================================================================" -ForegroundColor Cyan
}

# Main Selection
if ($Stage -eq "1") { Run-Stage1 }
elseif ($Stage -eq "2") { Run-Stage2 }
elseif ($Stage -eq "3") { Run-Stage3 }
elseif ($Stage -eq "4") { Run-Stage4 }
elseif ($Stage -eq "all") { Run-All }
else {
    Show-Header
    Write-Host "Select which stage to run:" -ForegroundColor White
    Write-Host "  [1] Stage 1: Competitor Baselines (Default ExtraTrees, Random, Optuna, FLAML) [~15-20 min]" -ForegroundColor White
    Write-Host "  [2] Stage 2: EvoSynth Scaled Runs (EvoSynth@300 & EvoSynth@500)      [~1 hour]" -ForegroundColor White
    Write-Host "  [3] Stage 3: Covertype Ablation CVGS-300 (All 5 Algorithms)         [Overnight]" -ForegroundColor White
    Write-Host "  [4] Stage 4: Deep Covertype Ablation CVGS-500 (All 5 Algorithms)    [Overnight]" -ForegroundColor White
    Write-Host "  [A] Run All Stages Sequentially (1 -> 2 -> 3 -> 4)                   [Overnight]" -ForegroundColor Yellow
    Write-Host "  [Q] Quit" -ForegroundColor Red
    Write-Host ""
    $choice = Read-Host "Enter option (1/2/3/4/A/Q)"

    switch ($choice.ToUpper()) {
        "1" { Run-Stage1 }
        "2" { Run-Stage2 }
        "3" { Run-Stage3 }
        "4" { Run-Stage4 }
        "A" { Run-All }
        default { Write-Host "Exiting." -ForegroundColor Gray }
    }
}
