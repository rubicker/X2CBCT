param(
    [string]$Root = "D:\code\X2CBCT\dataset_list\self_projection1208",
    [string]$CasePath = "",
    [int[]]$TargetAngles = @(200, 290),
    [switch]$AllCases
)

function Get-ProjectionFiles {
    param([string]$Path)

    Get-ChildItem -File -Path $Path |
        Where-Object { $_.Name -match '^(\d+)_(-?\d+)\.nii\.gz$' } |
        ForEach-Object {
            [void]($_.Name -match '^(\d+)_(-?\d+)\.nii\.gz$')
            [pscustomobject]@{
                Name       = $_.Name
                Index      = [int]$Matches[1]
                AngleLabel = [int]$Matches[2]
                FullName   = $_.FullName
            }
        } |
        Sort-Object Index
}

function Get-CaseGeometry {
    param([string]$Path)

    $geometryPath = Join-Path $Path "Geometry"
    if (-not (Test-Path -LiteralPath $geometryPath)) {
        throw "Missing Geometry: $geometryPath"
    }

    [xml]$xml = Get-Content -Raw -Encoding UTF8 -Path $geometryPath
    $projections = @($xml.RTKThreeDCircularGeometry.Projection)
    $angles = $projections | ForEach-Object { [double]$_.GantryAngle }

    [pscustomobject]@{
        SourceToIsocenterDistance = [double]$xml.RTKThreeDCircularGeometry.SourceToIsocenterDistance
        SourceToDetectorDistance  = [double]$xml.RTKThreeDCircularGeometry.SourceToDetectorDistance
        ProjectionCount           = $projections.Count
        Angles                    = $angles
    }
}

function Get-CircularDistance {
    param([double]$A, [double]$B)

    $diff = [math]::Abs($A - $B) % 360
    if ($diff -gt 180) { return 360 - $diff }
    return $diff
}

function Summarize-Case {
    param([string]$Path)

    $files = @(Get-ProjectionFiles -Path $Path)
    $geometry = Get-CaseGeometry -Path $Path
    $pairCount = [math]::Min($files.Count, $geometry.ProjectionCount)
    $pairs = for ($i = 0; $i -lt $pairCount; $i++) {
        $angle = [double]$geometry.Angles[$i]
        $floorAngle = [math]::Floor(($angle + 360) % 360)
        [pscustomobject]@{
            Ordinal    = $i + 1
            File       = $files[$i].Name
            FileIndex  = $files[$i].Index
            FileLabel  = $files[$i].AngleLabel
            Gantry     = $angle
            FloorAngle = [int]$floorAngle
            LabelMatch = ($files[$i].AngleLabel -eq [int]$floorAngle)
        }
    }

    $mismatches = @($pairs | Where-Object { -not $_.LabelMatch })
    $nearest = foreach ($target in $TargetAngles) {
        $pairs |
            Sort-Object @{ Expression = { Get-CircularDistance -A $_.Gantry -B $target } } |
            Select-Object -First 1 -Property @{ Name = "Target"; Expression = { $target } }, File, Ordinal, FileIndex, FileLabel, Gantry, @{ Name = "AbsDiffDeg"; Expression = { [math]::Round((Get-CircularDistance -A $_.Gantry -B $target), 6) } }
    }

    [pscustomobject]@{
        CasePath                 = $Path
        ProjectionFileCount      = $files.Count
        GeometryProjectionCount  = $geometry.ProjectionCount
        SID                      = $geometry.SourceToIsocenterDistance
        SDD                      = $geometry.SourceToDetectorDistance
        MinGantry                = [math]::Round((($geometry.Angles | Measure-Object -Minimum).Minimum), 6)
        MaxGantry                = [math]::Round((($geometry.Angles | Measure-Object -Maximum).Maximum), 6)
        FirstPairs               = @($pairs | Select-Object -First 8)
        LastPairs                = @($pairs | Select-Object -Last 8)
        LabelMismatchCount       = $mismatches.Count
        FirstLabelMismatches     = @($mismatches | Select-Object -First 8)
        NearestTargetProjections = @($nearest)
    }
}

if ($CasePath) {
    Summarize-Case -Path $CasePath | ConvertTo-Json -Depth 6
    exit
}

if ($AllCases) {
    $caseDirs = Get-ChildItem -Directory -Path $Root |
        ForEach-Object { Get-ChildItem -Directory -Path $_.FullName } |
        Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName "Geometry") }

    $rows = foreach ($case in $caseDirs) {
        $summary = Summarize-Case -Path $case.FullName
        [pscustomobject]@{
            Patient                  = Split-Path (Split-Path $case.FullName -Parent) -Leaf
            Case                     = Split-Path $case.FullName -Leaf
            ProjectionFileCount      = $summary.ProjectionFileCount
            GeometryProjectionCount  = $summary.GeometryProjectionCount
            SID                      = $summary.SID
            SDD                      = $summary.SDD
            MinGantry                = $summary.MinGantry
            MaxGantry                = $summary.MaxGantry
            LabelMismatchCount       = $summary.LabelMismatchCount
        }
    }

    $rows | ConvertTo-Json -Depth 3
    exit
}

throw "Pass -CasePath for one case or -AllCases for dataset summary."
