param(
    [int]$Port = 8765,
    [int]$TargetStep = 1500,
    [string]$LogPath = "metrics\foundational_recovery_v2_probe500.log"
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repo

$htmlPath = Join-Path $PSScriptRoot "training_dashboard.html"
$fullLogPath = Join-Path $repo $LogPath

$proofPath = Join-Path $repo `
    "metrics\foundational_recovery_v2_probe500_gate_after_stop_fix.json"

$memorizationPath = Join-Path $repo `
    "metrics\foundational_mem32_generation.json"

function Convert-ToDoubleOrNull {
    param($Value)

    if ($null -eq $Value) {
        return $null
    }

    return [double]$Value
}

function Read-JsonFileSafe {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        return $null
    }

    try {
        return Get-Content $Path -Raw | ConvertFrom-Json
    }
    catch {
        return $null
    }
}

function Get-TrainingStatus {
    $points = [System.Collections.Generic.List[object]]::new()

    if (Test-Path $fullLogPath) {
        foreach ($line in [System.IO.File]::ReadAllLines($fullLogPath)) {
            if ([string]::IsNullOrWhiteSpace($line)) {
                continue
            }

            # Python emitted Infinity in the first start event.
            # Convert it to valid JSON null before parsing.
            $cleanLine = $line -replace '\bInfinity\b', 'null'

            try {
                $event = $cleanLine | ConvertFrom-Json
            }
            catch {
                continue
            }

            if ($null -ne $event.sft_step) {
                $points.Add($event)
            }
        }
    }

    $current = $null
    $best = $null

    if ($points.Count -gt 0) {
        $current = $points[$points.Count - 1]

        $best = $points |
            Where-Object { $null -ne $_.val_loss } |
            Sort-Object { [double]$_.val_loss } |
            Select-Object -First 1
    }

    $currentStep = if ($null -ne $current) {
        [int]$current.sft_step
    }
    else {
        0
    }

    $secondsPerStep = $null

    if ($points.Count -ge 2) {
        $previous = $points[$points.Count - 2]
        $latest = $points[$points.Count - 1]

        $stepDifference =
            [double]$latest.sft_step - [double]$previous.sft_step

        $timeDifference =
            [double]$latest.time - [double]$previous.time

        if ($stepDifference -gt 0 -and $timeDifference -gt 0) {
            $secondsPerStep = $timeDifference / $stepDifference
        }
    }

    $remainingSteps = [math]::Max(0, $TargetStep - $currentStep)

    $etaSeconds = if ($null -ne $secondsPerStep) {
        $secondsPerStep * $remainingSteps
    }
    else {
        $null
    }

    $trainerProcesses = @(
        Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
            Where-Object {
                $_.CommandLine -match 'train_forgeloop_sft\.py' -and
                $_.CommandLine -match 'foundational_recovery_v2_probe500'
            }
    )

    $running = $trainerProcesses.Count -gt 0

    $proofReport = Read-JsonFileSafe $proofPath
    $memorizationReport = Read-JsonFileSafe $memorizationPath

    $proof = if ($null -ne $proofReport) {
        [ordered]@{
            passed = [bool]$proofReport.gate_passed
            score  = [double]$proofReport.overall_score
        }
    }
    else {
        $null
    }

    $memorization = if ($null -ne $memorizationReport) {
        [ordered]@{
            passed  = [bool]$memorizationReport.diagnostic_passed
            matches = [int]$memorizationReport.exact_matches
            total   = [int]$memorizationReport.total
        }
    }
    else {
        $null
    }

    $chartPoints = @(
        $points | ForEach-Object {
            [ordered]@{
                step      = [int]$_.sft_step
                trainLoss = Convert-ToDoubleOrNull $_.train_loss
                valLoss   = Convert-ToDoubleOrNull $_.val_loss
            }
        }
    )

    return [ordered]@{
        currentStep    = $currentStep
        targetStep     = $TargetStep
        progress       = if ($TargetStep -gt 0) {
            [math]::Min(100, 100 * $currentStep / $TargetStep)
        }
        else {
            0
        }

        trainLoss      = if ($null -ne $current) {
            Convert-ToDoubleOrNull $current.train_loss
        }
        else {
            $null
        }

        valLoss        = if ($null -ne $current) {
            Convert-ToDoubleOrNull $current.val_loss
        }
        else {
            $null
        }

        bestValLoss    = if ($null -ne $best) {
            Convert-ToDoubleOrNull $best.val_loss
        }
        else {
            $null
        }

        bestStep       = if ($null -ne $best) {
            [int]$best.sft_step
        }
        else {
            $null
        }

        gradNorm       = if ($null -ne $current) {
            Convert-ToDoubleOrNull $current.grad_norm
        }
        else {
            $null
        }

        bytesPerSecond = if ($null -ne $current) {
            Convert-ToDoubleOrNull $current.bytes_per_second
        }
        else {
            $null
        }

        badEvals       = if ($null -ne $current) {
            [int]$current.bad_evals
        }
        else {
            $null
        }

        improved       = if ($null -ne $current) {
            [bool]$current.improved
        }
        else {
            $null
        }

        sampledDomains = if ($null -ne $current) {
            @($current.sampled_domains)
        }
        else {
            @()
        }

        running        = $running
        etaSeconds     = $etaSeconds
        proof          = $proof
        memorization   = $memorization
        points         = $chartPoints
        updatedAt      = (Get-Date).ToString("o")
    }
}

function Write-HttpResponse {
    param(
        [System.Net.Sockets.NetworkStream]$Stream,
        [int]$StatusCode,
        [string]$ContentType,
        [string]$Content
    )

    $statusText = if ($StatusCode -eq 200) {
        "OK"
    }
    elseif ($StatusCode -eq 204) {
        "No Content"
    }
    else {
        "Not Found"
    }

    $contentBytes = [System.Text.Encoding]::UTF8.GetBytes($Content)

    $header = @(
        "HTTP/1.1 $StatusCode $statusText"
        "Content-Type: $ContentType; charset=utf-8"
        "Content-Length: $($contentBytes.Length)"
        "Cache-Control: no-store, no-cache, must-revalidate"
        "Connection: close"
        ""
        ""
    ) -join "`r`n"

    $headerBytes = [System.Text.Encoding]::ASCII.GetBytes($header)

    $Stream.Write($headerBytes, 0, $headerBytes.Length)

    if ($contentBytes.Length -gt 0) {
        $Stream.Write($contentBytes, 0, $contentBytes.Length)
    }

    $Stream.Flush()
}

if (-not (Test-Path $htmlPath)) {
    throw "Dashboard HTML not found: $htmlPath"
}

$server = [System.Net.Sockets.TcpListener]::new(
    [System.Net.IPAddress]::Loopback,
    $Port
)

$server.Start()

$url = "http://127.0.0.1:$Port/"

Write-Host ""
Write-Host "NUERONCE training dashboard is running:" -ForegroundColor Cyan
Write-Host $url -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C in this window to stop only the dashboard."
Write-Host "The training process will not be stopped."
Write-Host ""

Start-Process $url

try {
    while ($true) {
        $client = $server.AcceptTcpClient()

        try {
            $stream = $client.GetStream()

            $reader = [System.IO.StreamReader]::new(
                $stream,
                [System.Text.Encoding]::ASCII,
                $false,
                1024,
                $true
            )

            $requestLine = $reader.ReadLine()

            while ($true) {
                $headerLine = $reader.ReadLine()

                if ($null -eq $headerLine -or $headerLine -eq "") {
                    break
                }
            }

            $requestParts = $requestLine -split " "
            $requestPath = if ($requestParts.Count -ge 2) {
                ($requestParts[1] -split "\?")[0]
            }
            else {
                "/"
            }

            switch ($requestPath) {
                "/api/status" {
                    $json = Get-TrainingStatus |
                        ConvertTo-Json -Depth 12 -Compress

                    Write-HttpResponse `
                        -Stream $stream `
                        -StatusCode 200 `
                        -ContentType "application/json" `
                        -Content $json
                }

                "/favicon.ico" {
                    Write-HttpResponse `
                        -Stream $stream `
                        -StatusCode 204 `
                        -ContentType "text/plain" `
                        -Content ""
                }

                "/" {
                    $html = Get-Content $htmlPath -Raw

                    Write-HttpResponse `
                        -Stream $stream `
                        -StatusCode 200 `
                        -ContentType "text/html" `
                        -Content $html
                }

                default {
                    Write-HttpResponse `
                        -Stream $stream `
                        -StatusCode 404 `
                        -ContentType "text/plain" `
                        -Content "Not found"
                }
            }
        }
        catch {
            Write-Warning $_
        }
        finally {
            $client.Close()
        }
    }
}
finally {
    $server.Stop()
}
