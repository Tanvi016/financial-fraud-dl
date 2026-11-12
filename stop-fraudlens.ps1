$runtime = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) 'runtime'
foreach ($name in @('api','frontend')) {
    $pidFile = Join-Path $runtime "$name.pid"
    if (Test-Path -LiteralPath $pidFile) {
        $processId = [int](Get-Content -LiteralPath $pidFile -Raw)
        $allProcesses = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue
        $rootProcess = $allProcesses | Where-Object { $_.ProcessId -eq $processId } | Select-Object -First 1
        $expectedCommand = if ($name -eq 'api') { 'uvicorn backend.app:app' } else { 'vite.cmd" preview' }
        if ($rootProcess -and $rootProcess.CommandLine -notlike "*$expectedCommand*") {
            Write-Warning "Skipping PID $processId because it no longer matches the FraudLens $name launcher."
            Remove-Item -LiteralPath $pidFile -Force
            continue
        }
        $pending = [System.Collections.Generic.Queue[int]]::new()
        $pending.Enqueue($processId)
        $descendants = [System.Collections.Generic.List[int]]::new()
        while ($pending.Count -gt 0) {
            $parentId = $pending.Dequeue()
            foreach ($child in $allProcesses | Where-Object { $_.ParentProcessId -eq $parentId }) {
                $descendants.Add([int]$child.ProcessId)
                $pending.Enqueue([int]$child.ProcessId)
            }
        }
        $descendantIds = $descendants.ToArray()
        for ($index = $descendantIds.Length - 1; $index -ge 0; $index--) {
            Stop-Process -Id $descendantIds[$index] -Force -ErrorAction SilentlyContinue
        }
        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $pidFile -Force
    }
}
Write-Output 'FraudLens launcher processes stopped.'
