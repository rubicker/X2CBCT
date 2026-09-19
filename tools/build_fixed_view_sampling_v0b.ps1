param(
    [string]$Root = "D:\code\X2CBCT\dataset_list\self_projection1208",
    [string]$OutputDir = "D:\code\X2CBCT\data_manifest",
    [double]$MaxAngularResidualDeg = 3.0
)

$TargetSets = [ordered]@{
    view_2  = @(200, 290)
    view_5  = @(200, 230, 260, 290, 320)
    view_10 = @(200, 215, 230, 245, 260, 275, 290, 305, 320, 335)
}

function Get-CircularDistance {
    param([double]$A, [double]$B)

    $diff = [math]::Abs($A - $B) % 360
    if ($diff -gt 180) { return 360 - $diff }
    return $diff
}

function Get-ProjectionFiles {
    param([string]$CasePath)

    Get-ChildItem -File -Path $CasePath |
        Where-Object { $_.Name -match '^(\d+)_(-?\d+)\.nii\.gz$' } |
        ForEach-Object {
            [void]($_.Name -match '^(\d+)_(-?\d+)\.nii\.gz$')
            [pscustomobject]@{
                File       = $_.Name
                FileIndex  = [int]$Matches[1]
                AngleLabel = [int]$Matches[2]
                Path       = $_.FullName
            }
        } |
        Sort-Object FileIndex
}

function Get-GeometryAngles {
    param([string]$CasePath)

    $geometryPath = Join-Path $CasePath "Geometry"
    [xml]$xml = Get-Content -Raw -Encoding UTF8 -Path $geometryPath
    @($xml.RTKThreeDCircularGeometry.Projection | ForEach-Object { [double]$_.GantryAngle })
}

function Select-NearestProjection {
    param(
        [array]$Files,
        [array]$Angles,
        [double]$TargetAngle
    )

    $bestIndex = -1
    $bestDistance = [double]::PositiveInfinity

    for ($i = 0; $i -lt $Angles.Count; $i++) {
        $distance = Get-CircularDistance -A ([double]$Angles[$i]) -B $TargetAngle
        if ($distance -lt $bestDistance) {
            $bestDistance = $distance
            $bestIndex = $i
        }
    }

    [pscustomobject]@{
        target_angle_deg       = $TargetAngle
        ordinal                = $bestIndex + 1
        file                   = $Files[$bestIndex].File
        file_index             = $Files[$bestIndex].FileIndex
        file_angle_label       = $Files[$bestIndex].AngleLabel
        gantry_angle_deg       = [math]::Round([double]$Angles[$bestIndex], 9)
        angular_residual_deg   = [math]::Round($bestDistance, 9)
    }
}

function Convert-TargetSetsForJson {
    param([hashtable]$TargetSets)

    $obj = [ordered]@{}
    foreach ($key in $TargetSets.Keys) {
        $obj[$key] = @($TargetSets[$key])
    }
    $obj
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$caseDirs = Get-ChildItem -Directory -Path $Root |
    ForEach-Object { Get-ChildItem -Directory -Path $_.FullName } |
    Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName "Geometry") } |
    Sort-Object FullName

$records = @()
$excluded = @()

foreach ($caseDir in $caseDirs) {
    $patientId = Split-Path (Split-Path $caseDir.FullName -Parent) -Leaf
    $caseId = Split-Path $caseDir.FullName -Leaf
    $files = @(Get-ProjectionFiles -CasePath $caseDir.FullName)
    $angles = @(Get-GeometryAngles -CasePath $caseDir.FullName)

    $selectedSets = [ordered]@{}
    $maxResidual = 0.0
    $failedTargets = @()
    $duplicateSelections = @()

    foreach ($setName in $TargetSets.Keys) {
        $selected = @()
        foreach ($target in $TargetSets[$setName]) {
            $pick = Select-NearestProjection -Files $files -Angles $angles -TargetAngle $target
            $selected += $pick
            if ($pick.angular_residual_deg -gt $maxResidual) {
                $maxResidual = $pick.angular_residual_deg
            }
            if ($pick.angular_residual_deg -gt $MaxAngularResidualDeg) {
                $failedTargets += [pscustomobject]@{
                    view_set             = $setName
                    target_angle_deg     = $target
                    best_file            = $pick.file
                    best_gantry_angle    = $pick.gantry_angle_deg
                    angular_residual_deg = $pick.angular_residual_deg
                }
            }
        }

        $duplicateOrdinals = @($selected | Group-Object ordinal | Where-Object { $_.Count -gt 1 })
        foreach ($dup in $duplicateOrdinals) {
            $duplicateSelections += [pscustomobject]@{
                view_set = $setName
                ordinal  = [int]$dup.Name
                count    = $dup.Count
            }
        }
        $selectedSets[$setName] = $selected
    }

    if ($failedTargets.Count -gt 0 -or $duplicateSelections.Count -gt 0 -or $files.Count -ne $angles.Count) {
        $excluded += [pscustomobject]@{
            patient_id              = $patientId
            case_id                 = $caseId
            case_path               = $caseDir.FullName
            projection_file_count   = $files.Count
            geometry_projection_count = $angles.Count
            max_residual_deg        = [math]::Round($maxResidual, 9)
            failed_targets          = $failedTargets
            duplicate_selections    = $duplicateSelections
        }
        continue
    }

    $records += [pscustomobject]@{
        patient_id              = $patientId
        case_id                 = $caseId
        case_path               = $caseDir.FullName
        projection_file_count   = $files.Count
        geometry_projection_count = $angles.Count
        max_residual_deg        = [math]::Round($maxResidual, 9)
        selected                = $selectedSets
    }
}

$manifest = [ordered]@{
    generated_at                 = (Get-Date).ToString("o")
    root                         = $Root
    protocol                     = "v0b_fixed_view_sampling_rtk_gantry_nearest_neighbor"
    max_angular_residual_deg      = $MaxAngularResidualDeg
    target_sets                  = (Convert-TargetSetsForJson -TargetSets $TargetSets)
    included_case_count          = $records.Count
    excluded_case_count          = $excluded.Count
    records                      = $records
    excluded_cases               = $excluded
}

$summary = [ordered]@{
    generated_at                 = $manifest.generated_at
    root                         = $Root
    target_sets                  = $manifest.target_sets
    max_angular_residual_deg      = $MaxAngularResidualDeg
    total_case_count             = $caseDirs.Count
    included_case_count          = $records.Count
    excluded_case_count          = $excluded.Count
    included_patient_count       = @($records | Select-Object -ExpandProperty patient_id -Unique).Count
    excluded_cases               = @($excluded | Select-Object patient_id, case_id, projection_file_count, geometry_projection_count, max_residual_deg, failed_targets)
}

$manifestPath = Join-Path $OutputDir "fixed_view_sampling_v0b.json"
$summaryPath = Join-Path $OutputDir "fixed_view_sampling_v0b_summary.json"

$manifest | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 -Path $manifestPath
$summary | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 -Path $summaryPath

[pscustomobject]@{
    ManifestPath       = $manifestPath
    SummaryPath        = $summaryPath
    IncludedCaseCount  = $records.Count
    ExcludedCaseCount  = $excluded.Count
    IncludedPatientCount = $summary.included_patient_count
} | ConvertTo-Json -Depth 3
