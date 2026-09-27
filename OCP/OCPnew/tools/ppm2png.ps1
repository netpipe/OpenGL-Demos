param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$Out,
    [int]$Scale = 1
)

Add-Type -AssemblyName System.Drawing

$fs = [System.IO.File]::OpenRead($Path)
$br = New-Object System.IO.BinaryReader $fs

function Read-Token {
    $sb = New-Object System.Text.StringBuilder
    while ($true) {
        $b = $br.ReadByte()
        $c = [char]$b
        if ($c -eq '#') { while ([char]$br.ReadByte() -ne "`n") {} ; continue }
        if ($c -match '\s') { if ($sb.Length -gt 0) { break } else { continue } }
        [void]$sb.Append($c)
    }
    return $sb.ToString()
}

$magic = Read-Token
if ($magic -ne "P6" -and $magic -ne "P5") { throw "not a binary PPM/PGM: $magic" }
$w = [int](Read-Token)
$h = [int](Read-Token)
$null = Read-Token   # maxval

if ($magic -eq "P6") {
    $data = $br.ReadBytes($w * $h * 3)
} else {
    $grey = $br.ReadBytes($w * $h)
    $data = New-Object byte[] ($w * $h * 3)
    for ($i = 0; $i -lt $grey.Length; $i++) {
        $data[$i * 3] = $grey[$i]; $data[$i * 3 + 1] = $grey[$i]; $data[$i * 3 + 2] = $grey[$i]
    }
}
$br.Close(); $fs.Close()

$bmp = New-Object System.Drawing.Bitmap $w, $h, ([System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
$rect = New-Object System.Drawing.Rectangle 0, 0, $w, $h
$bd = $bmp.LockBits($rect, [System.Drawing.Imaging.ImageLockMode]::WriteOnly, [System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
$stride = $bd.Stride
$row = New-Object byte[] $stride
for ($y = 0; $y -lt $h; $y++) {
    for ($x = 0; $x -lt $w; $x++) {
        $s = ($y * $w + $x) * 3
        $row[$x * 3 + 0] = $data[$s + 2]  # B
        $row[$x * 3 + 1] = $data[$s + 1]  # G
        $row[$x * 3 + 2] = $data[$s + 0]  # R
    }
    [System.Runtime.InteropServices.Marshal]::Copy($row, 0, [IntPtr]::Add($bd.Scan0, $y * $stride), $stride)
}
$bmp.UnlockBits($bd)

if ($Scale -gt 1) {
    $big = New-Object System.Drawing.Bitmap ($w * $Scale), ($h * $Scale)
    $g = [System.Drawing.Graphics]::FromImage($big)
    $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::NearestNeighbor
    $g.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::Half
    $g.DrawImage($bmp, 0, 0, ($w * $Scale), ($h * $Scale))
    $g.Dispose()
    $bmp.Dispose()
    $bmp = $big
}

$bmp.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
$bmp.Dispose()
Write-Output "wrote $Out"
