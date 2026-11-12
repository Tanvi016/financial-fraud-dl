$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$runtime = Join-Path $root 'runtime'
$null = New-Item -ItemType Directory -Force -Path $runtime
$python = Join-Path $root '.venv\Scripts\python.exe'
$vite = Join-Path $root 'frontend\node_modules\.bin\vite.cmd'

if (-not (Test-Path -LiteralPath $python)) { throw 'Project Python environment is missing. Follow the README installation steps first.' }
if (-not (Test-Path -LiteralPath $vite)) { throw 'Frontend dependencies are missing. Run pnpm install from the frontend folder first.' }
if (-not (Test-Path -LiteralPath (Join-Path $root 'frontend\dist\index.html'))) { throw 'Frontend build is missing. Run pnpm build from the frontend folder first.' }

function Test-ListeningPort([int]$Port) {
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $client.Connect('127.0.0.1', $Port)
        return $true
    } catch {
        return $false
    } finally {
        $client.Dispose()
    }
}

if (-not (Test-ListeningPort 8000)) {
    $api = Start-Process -FilePath $python -ArgumentList @('-m','uvicorn','backend.app:app','--host','127.0.0.1','--port','8000') `
        -WorkingDirectory $root -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $runtime 'api.log') -RedirectStandardError (Join-Path $runtime 'api-error.log')
    Set-Content -LiteralPath (Join-Path $runtime 'api.pid') -Value $api.Id
}

if (-not (Test-ListeningPort 5173)) {
    $command = '"' + $vite + '" preview --host 127.0.0.1 --port 5173'
    $web = Start-Process -FilePath $env:ComSpec -ArgumentList @('/c', $command) `
        -WorkingDirectory (Join-Path $root 'frontend') -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $runtime 'frontend.log') -RedirectStandardError (Join-Path $runtime 'frontend-error.log')
    Set-Content -LiteralPath (Join-Path $runtime 'frontend.pid') -Value $web.Id
}

$health = $null
for ($attempt = 0; $attempt -lt 45 -and -not $health; $attempt++) {
    try {
        $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/health' -TimeoutSec 2
    } catch {
        Start-Sleep -Seconds 1
    }
}
if ($health) {
    Write-Output "FraudLens API is $($health.status); models_ready=$($health.models_ready). ANN is warmed for faster scoring."
} else {
    Write-Warning "API did not become ready. See $runtime\api-error.log"
}
Write-Output 'Dashboard: http://127.0.0.1:5173/'
