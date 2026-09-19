param(
    [string]$ManifestPath = "D:\code\X2CBCT\data_manifest\fixed_view_sampling_v0b.json",
    [string]$OutputDir = "D:\code\X2CBCT\data_manifest",
    [double]$TestRatio = 0.20,
    [int]$Seed = 20260517
)

function Get-StableHash {
    param([string]$Text)

    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($Text)
        $hash = $sha.ComputeHash($bytes)
        return -join ($hash | ForEach-Object { $_.ToString("x2") })
    }
    finally {
        $sha.Dispose()
    }
}

function Get-Combinations {
    param(
        [string[]]$Items,
        [int]$Choose,
        [int]$Start = 0,
        [string[]]$Prefix = @()
    )

    if ($Prefix.Count -eq $Choose) {
        ,$Prefix
        return
    }

    $remainingNeeded = $Choose - $Prefix.Count
    for ($i = $Start; $i -le $Items.Count - $remainingNeeded; $i++) {
        Get-Combinations -Items $Items -Choose $Choose -Start ($i + 1) -Prefix ($Prefix + $Items[$i])
    }
}

if (-not (Test-Path -LiteralPath $ManifestPath)) {
    throw "Missing manifest: $ManifestPath"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$manifest = Get-Content -Raw -Encoding UTF8 -Path $ManifestPath | ConvertFrom-Json
$records = @($manifest.records)
if ($records.Count -eq 0) {
    throw "No included records found in manifest: $ManifestPath"
}

$patientGroups = @($records | Group-Object patient_id | Sort-Object { [int]$_.Name })
$patients = @($patientGroups | ForEach-Object { $_.Name })
$patientCaseCounts = @{}
foreach ($group in $patientGroups) {
    $patientCaseCounts[$group.Name] = $group.Count
}

$totalCases = $records.Count
$targetTestCases = [math]::Round($totalCases * $TestRatio)
$targetTestPatients = [math]::Max(1, [math]::Round($patients.Count * $TestRatio))

$best = $null
foreach ($combo in Get-Combinations -Items $patients -Choose $targetTestPatients) {
    $caseCount = 0
    foreach ($patient in $combo) {
        $caseCount += [int]$patientCaseCounts[$patient]
    }

    $caseDiff = [math]::Abs($caseCount - $targetTestCases)
    $hash = Get-StableHash -Text ("$Seed|" + (($combo | Sort-Object { [int]$_ }) -join ","))
    $candidate = [pscustomobject]@{
        Patients  = @($combo | Sort-Object { [int]$_ })
        CaseCount = $caseCount
        CaseDiff  = $caseDiff
        Hash      = $hash
    }

    if ($null -eq $best -or
        $candidate.CaseDiff -lt $best.CaseDiff -or
        ($candidate.CaseDiff -eq $best.CaseDiff -and $candidate.Hash -lt $best.Hash)) {
        $best = $candidate
    }
}

$testPatients = @($best.Patients)
$trainPatients = @($patients | Where-Object { $testPatients -notcontains $_ } | Sort-Object { [int]$_ })

$trainRecords = @($records | Where-Object { $trainPatients -contains $_.patient_id })
$testRecords = @($records | Where-Object { $testPatients -contains $_.patient_id })

$trainCases = @($trainRecords | ForEach-Object { [pscustomobject]@{ patient_id = $_.patient_id; case_id = $_.case_id; case_path = $_.case_path } })
$testCases = @($testRecords | ForEach-Object { [pscustomobject]@{ patient_id = $_.patient_id; case_id = $_.case_id; case_path = $_.case_path } })

$split = [ordered]@{
    generated_at                  = (Get-Date).ToString("o")
    protocol                      = "v0b_patient_level_train_test_no_val"
    source_manifest               = $ManifestPath
    seed                          = $Seed
    test_ratio                    = $TestRatio
    note                          = "V0b uses train/test only. Validation is intentionally omitted; model selection should rely on train sanity checks and test should remain held out."
    train                         = [ordered]@{
        patient_count = $trainPatients.Count
        case_count    = $trainRecords.Count
        patients      = $trainPatients
        cases         = $trainCases
    }
    test                          = [ordered]@{
        patient_count = $testPatients.Count
        case_count    = $testRecords.Count
        patients      = $testPatients
        cases         = $testCases
    }
    leakage_checks                = [ordered]@{
        patient_overlap_count = @($trainPatients | Where-Object { $testPatients -contains $_ }).Count
        case_overlap_count    = @($trainCases | ForEach-Object { "$($_.patient_id)/$($_.case_id)" } | Where-Object {
            $testCaseIds = @($testCases | ForEach-Object { "$($_.patient_id)/$($_.case_id)" })
            $testCaseIds -contains $_
        }).Count
    }
}

$summary = [ordered]@{
    generated_at           = $split.generated_at
    protocol               = $split.protocol
    source_manifest        = $ManifestPath
    seed                   = $Seed
    test_ratio             = $TestRatio
    total_patient_count    = $patients.Count
    total_case_count       = $totalCases
    target_test_patients   = $targetTestPatients
    target_test_cases      = $targetTestCases
    train_patient_count    = $trainPatients.Count
    train_case_count       = $trainRecords.Count
    train_patients         = $trainPatients
    test_patient_count     = $testPatients.Count
    test_case_count        = $testRecords.Count
    test_patients          = $testPatients
    patient_overlap_count  = $split.leakage_checks.patient_overlap_count
    case_overlap_count     = $split.leakage_checks.case_overlap_count
}

$splitPath = Join-Path $OutputDir "splits_v0b_train_test.json"
$summaryPath = Join-Path $OutputDir "splits_v0b_train_test_summary.json"

$split | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 -Path $splitPath
$summary | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 -Path $summaryPath

[pscustomobject]@{
    SplitPath           = $splitPath
    SummaryPath         = $summaryPath
    TrainPatientCount   = $trainPatients.Count
    TrainCaseCount      = $trainRecords.Count
    TestPatientCount    = $testPatients.Count
    TestCaseCount       = $testRecords.Count
    PatientOverlapCount = $split.leakage_checks.patient_overlap_count
    CaseOverlapCount    = $split.leakage_checks.case_overlap_count
} | ConvertTo-Json -Depth 3
