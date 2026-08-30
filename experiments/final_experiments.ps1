param(
    [ValidateSet('E1_oracle','E2_throughput','E3_ba128','E3_waxman192')]
    [string[]]$Cases = @('E1_oracle','E2_throughput','E3_ba128','E3_waxman192'),
    [int]$Seeds = 20
)

$ErrorActionPreference = 'Stop'
$Checkpoint = 'results\server_generalization_v2\20260815_072912_pid1297\models\seed_20260821\online_milp_gnn.pt'
$Root = 'results\final_experiments'

function Invoke-Comparison {
    param(
        [string]$Name,
        [string]$Profile,
        [int]$SeedStart,
        [hashtable]$Options
    )
    $Output = Join-Path $Root $Name
    $Cmd = @(
        'python','-m','algorithms.telgen.compare_online_gnn',
        '--checkpoint', $Checkpoint,
        '--output', $Output,
        '--seeds', $Seeds,
        '--seed-start', $SeedStart,
        '--comparison-profile', $Profile,
        '--gnn-device','cpu'
    )
    foreach ($Key in $Options.Keys) {
        $Cmd += @("--$($Key.Replace('_','-'))", $Options[$Key])
    }
    Write-Host "==== $Name ===="
    & $Cmd[0] $Cmd[1..($Cmd.Count-1)]
    if ($LASTEXITCODE -ne 0) {
        throw "experiment $Name failed with exit code $LASTEXITCODE"
    }
}

if ($Cases -contains 'E1_oracle') {
    Invoke-Comparison -Name 'E1_oracle' -Profile 'formal' -SeedStart 71000 -Options @{
        requests = 20; requests_per_batch = 5; nodes = 64; min_hops = 4; max_hops = 4;
        paths = 4; construction_plans = 5; ttl = 16;
        generation_probability = 1.0; swap_probability = 1.0; memory_capacity = 1
    }
}

if ($Cases -contains 'E2_throughput') {
    Invoke-Comparison -Name 'E2_throughput' -Profile 'scalable' -SeedStart 72000 -Options @{
        requests = 250; requests_per_batch = 25; nodes = 64; min_hops = 4; max_hops = 4;
        paths = 4; construction_plans = 5; ttl = 16;
        generation_probability = 1.0; swap_probability = 1.0; memory_capacity = 1
    }
}

if ($Cases -contains 'E3_ba128') {
    Invoke-Comparison -Name 'E3_ba128' -Profile 'scalable' -SeedStart 73000 -Options @{
        requests = 50; requests_per_batch = 10; nodes = 128; min_hops = 3; max_hops = 4;
        paths = 4; construction_plans = 5; ttl = 20; topology_mode = 'barabasi_albert';
        generation_probability = 1.0; swap_probability = 1.0; memory_capacity = 1
    }
}

if ($Cases -contains 'E3_waxman192') {
    Invoke-Comparison -Name 'E3_waxman192' -Profile 'scalable' -SeedStart 74000 -Options @{
        requests = 60; requests_per_batch = 10; nodes = 192; min_hops = 3; max_hops = 5;
        paths = 4; construction_plans = 5; ttl = 24; topology_mode = 'waxman';
        generation_probability = 1.0; swap_probability = 1.0; memory_capacity = 1
    }
}
