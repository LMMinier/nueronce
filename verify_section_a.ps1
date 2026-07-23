$Repo = "C:\nueronce-claude-new-session-tkjr3b"
Set-Location $Repo

Write-Host "=== GIT STATUS ==="
git status --porcelain=v1

Write-Host "`n=== SECTION A FILE DIFFS ==="
git diff -- scripts/train_forgeloop_sft.py | Select-Object -First 100

Write-Host "`n=== TEST FILE DIFFS ==="
git diff -- tests/test_safe_interruption.py | Select-Object -First 50

Write-Host "`n=== UNMODIFIED FILES CHECK ==="
git diff --name-status -- nueronce/incremental.py
$status = $?
if (-not $status) {
    Write-Host "nueronce/incremental.py: NOT IN DIFF (correct)"
}
