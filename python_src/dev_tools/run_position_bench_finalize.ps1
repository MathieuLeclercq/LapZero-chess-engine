<#
Supervise la finalisation sur Windows, avec journaux et statut persistants.
Lancer ce script depuis une tache Windows ponctuelle pour survivre a la fermeture
de SSH (OpenSSH peut arreter les descendants de Start-Process a la deconnexion).
Les annotations existantes sont conservees. Aucune recurrence n'est necessaire.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Root,
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$Stockfish,
    [Parameter(Mandatory = $true)][string]$JobDirectory,
    [ValidateRange(1, 64)][int]$Workers = 8
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Quote-Argument([string]$Value) {
    if ($Value.Contains('"')) {
        throw 'Les chemins ne doivent pas contenir de guillemet.'
    }
    return '"' + $Value + '"'
}

foreach ($path in @($Root, $Python, $Stockfish, $JobDirectory)) {
    if (-not [IO.Path]::IsPathRooted($path)) {
        throw "Un chemin absolu est requis : $path"
    }
}
foreach ($path in @($Root, $Python, $Stockfish)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Chemin absent : $path"
    }
}
$builder = Join-Path $Root 'python_src\evaluation\build_position_bench.py'
$workDirectory = Join-Path $Root 'data\position_bench_work'
$outputDirectory = Join-Path $Root 'data\position_bench\v1'
if (-not (Test-Path -LiteralPath (Join-Path $workDirectory 'annotated.jsonl.zst'))) {
    throw 'La reserve annotee est absente. Aucun calcul lance.'
}
if (Test-Path -LiteralPath (Join-Path $outputDirectory 'manifest.json')) {
    throw 'Le banc v1 est deja publie. Aucun calcul lance.'
}
if (Test-Path -LiteralPath $JobDirectory) {
    throw 'Utiliser un nouveau dossier de journaux pour chaque lancement.'
}
New-Item -ItemType Directory -Path $JobDirectory | Out-Null
$statusPath = Join-Path $JobDirectory 'status.json'
$status = [ordered]@{
    state = 'starting'
    supervisor_pid = $PID
    python_pid = $null
    started_utc = [DateTime]::UtcNow.ToString('o')
    finished_utc = $null
    exit_code = $null
    work_directory = $workDirectory
    output_directory = $outputDirectory
}
$status | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
$exitCode = 1
try {
    $env:PYTHONUTF8 = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    $arguments = @(
        '-u', (Quote-Argument $builder), 'finalize', '--stage', 'final',
        '--stockfish', (Quote-Argument $Stockfish), '--workers', "$Workers",
        '--work-dir', (Quote-Argument $workDirectory),
        '--output', (Quote-Argument $outputDirectory), '--resume'
    )
    $child = Start-Process -FilePath $Python -ArgumentList $arguments `
        -WorkingDirectory $Root -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $JobDirectory 'stdout.log') `
        -RedirectStandardError (Join-Path $JobDirectory 'stderr.log')
    $status.python_pid = $child.Id
    $status.state = 'running'
    $status | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
    $null = $child.Handle
    $child.WaitForExit()
    $exitCode = $child.ExitCode
    $status.state = if ($exitCode -eq 0) { 'completed' } else { 'failed' }
} catch {
    $status.state = 'failed'
    $status['error'] = $_.Exception.Message
} finally {
    $status.finished_utc = [DateTime]::UtcNow.ToString('o')
    $status.exit_code = $exitCode
    $status | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
}
exit $exitCode
