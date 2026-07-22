param(
    [Parameter(Mandatory = $true)][int]$TrainingPid
)

$ErrorActionPreference = 'Stop'
$workspace = 'C:\nueronce-claude-new-session-tkjr3b'
Set-Location -LiteralPath $workspace
Wait-Process -Id $TrainingPid

$best = Join-Path $workspace 'runs\foundational_executor\latest_best.pt'
$destination = Join-Path $workspace 'checkpoints\engine\cfna_foundational_executor_best.pkl'
if (-not (Test-Path -LiteralPath $best)) {
    throw "best checkpoint missing: $best"
}
python -u scripts\convert_torch_checkpoint_to_engine.py $best $destination --lr 0.00005
if ($LASTEXITCODE -ne 0) {
    throw "checkpoint conversion failed with exit code $LASTEXITCODE"
}
Set-Content -LiteralPath 'runs\foundational_executor\finalized.txt' -Value (
    "finalized=" + [DateTime]::UtcNow.ToString('o') + "`nengine_checkpoint=" + $destination
)
