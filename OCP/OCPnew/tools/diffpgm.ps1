# Diff two binary-ish PGMs of the same size. Red = only in A, blue = only in B.
param(
    [Parameter(Mandatory = $true)][string]$A,
    [Parameter(Mandatory = $true)][string]$B,
    [string]$Out = "",
    [int]$Threshold = 128
)

function ReadPgm([string]$p) {
    $raw = [System.IO.File]::ReadAllBytes($p)
    $i = 0; $fields = @()
    while ($fields.Count -lt 4) {
        while ($i -lt $raw.Length -and [char]$raw[$i] -match '\s') { $i++ }
        if ([char]$raw[$i] -eq '#') { while ([char]$raw[$i] -ne "`n") { $i++ }; continue }
        $s = ""
        while ($i -lt $raw.Length -and [char]$raw[$i] -notmatch '\s') { $s += [char]$raw[$i]; $i++ }
        $fields += $s
    }
    $i++
    [pscustomobject]@{ W = [int]$fields[1]; H = [int]$fields[2]; Data = $raw[$i..($raw.Length - 1)] }
}

$ia = ReadPgm $A
$ib = ReadPgm $B
if ($ia.W -ne $ib.W -or $ia.H -ne $ib.H) { throw "size mismatch $($ia.W)x$($ia.H) vs $($ib.W)x$($ib.H)" }
$w = $ia.W; $h = $ia.H

$onlyA = 0; $onlyB = 0; $inkA = 0; $inkB = 0
$mask = New-Object byte[] ($w * $h)
for ($k = 0; $k -lt $w * $h; $k++) {
    $pa = $ia.Data[$k] -lt $Threshold
    $pb = $ib.Data[$k] -lt $Threshold
    if ($pa) { $inkA++ }
    if ($pb) { $inkB++ }
    if ($pa -and -not $pb) { $mask[$k] = 1; $onlyA++ }
    elseif ($pb -and -not $pa) { $mask[$k] = 2; $onlyB++ }
    elseif ($pa) { $mask[$k] = 3 }
}

# structural mismatch: disagreements with no agreeing neighbour of the other
# polarity within 1 px, i.e. not explained by an edge landing one pixel over
$struct = 0
for ($y = 0; $y -lt $h; $y++) {
    for ($x = 0; $x -lt $w; $x++) {
        $k = $y * $w + $x
        if ($mask[$k] -ne 1 -and $mask[$k] -ne 2) { continue }
        $near = $false
        for ($dy = -1; $dy -le 1 -and -not $near; $dy++) {
            for ($dx = -1; $dx -le 1; $dx++) {
                $nx = $x + $dx; $ny = $y + $dy
                if ($nx -lt 0 -or $nx -ge $w -or $ny -lt 0 -or $ny -ge $h) { continue }
                $nk = $ny * $w + $nx
                $want = if ($mask[$k] -eq 1) { 0 } else { 3 }
                if ($mask[$nk] -eq $want) { $near = $true; break }
            }
        }
        if (-not $near) { $struct++ }
    }
}

Write-Output "A=$A ink $inkA   B=$B ink $inkB"
Write-Output "only-A (red) $onlyA   only-B (blue) $onlyB   raw $($onlyA + $onlyB)   structural $struct"

if ($Out -ne "") {
    Add-Type -AssemblyName System.Drawing
    $bmp = New-Object System.Drawing.Bitmap $w, $h
    for ($y = 0; $y -lt $h; $y++) {
        for ($x = 0; $x -lt $w; $x++) {
            $c = switch ($mask[$y * $w + $x]) {
                1 { [System.Drawing.Color]::FromArgb(230, 30, 30) }
                2 { [System.Drawing.Color]::FromArgb(30, 80, 230) }
                3 { [System.Drawing.Color]::FromArgb(40, 40, 40) }
                default { [System.Drawing.Color]::FromArgb(245, 245, 245) }
            }
            $bmp.SetPixel($x, $y, $c)
        }
    }
    $bmp.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
    $bmp.Dispose()
    Write-Output "wrote $Out"
}
