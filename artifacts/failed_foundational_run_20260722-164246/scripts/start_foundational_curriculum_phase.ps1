param([Parameter(Mandatory = $true)][int]$PreviousTrainingPid)

$ErrorActionPreference = 'Stop'
$workspace = 'C:\nueronce-claude-new-session-tkjr3b'
Set-Location -LiteralPath $workspace
Wait-Process -Id $PreviousTrainingPid

$args = @(
    '-u', 'scripts\train_forgeloop_sft.py',
    '--base', 'runs\foundational_executor\latest_best.pt',
    '--train', 'data\foundational_curriculum_v1\train.jsonl',
    '--val', 'data\foundational_curriculum_v1\val.jsonl',
    '--out', 'runs\foundational_curriculum\latest.pt',
    '--system-file', 'runs\forgeloop\system_prompt.txt',
    '--batch', '4', '--max-len', '768', '--lr', '0.00002',
    '--eval-every', '25', '--eval-examples', '140',
    '--patience', '20', '--min-delta', '0.0005',
    '--checkpoint-every', '25', '--max-steps', '100000',
    '--execution-depth', '2', '--balanced-domain-sampling',
    '--reset-convergence', '--seed', '91'
)
New-Item -ItemType Directory -Force -Path 'runs\foundational_curriculum' | Out-Null
$process = Start-Process -FilePath python -ArgumentList $args -WorkingDirectory $workspace `
    -RedirectStandardOutput "$workspace\runs\foundational_curriculum\training.log" `
    -RedirectStandardError "$workspace\runs\foundational_curriculum\training.err.log" `
    -WindowStyle Hidden -PassThru
Set-Content -LiteralPath 'runs\foundational_curriculum\training.pid' -Value $process.Id
$process.WaitForExit()

# Loss convergence is not the finish line: record a model-only generation gate.
python -u scripts\eval_foundational_proof_gate.py `
    --checkpoint runs\foundational_curriculum\latest_best.pt `
    --output metrics\foundational_curriculum_proof_gate.json `
    --no-fail-exit *> runs\foundational_curriculum\proof_gate.log
