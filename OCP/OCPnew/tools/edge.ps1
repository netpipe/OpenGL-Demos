param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][int]$Y,
    [int]$X0 = 0, [int]$X1 = -1,
    [switch]$Vertical
)

Add-Type -AssemblyName System.Drawing
$bmp = New-Object System.Drawing.Bitmap $Path
$w = $bmp.Width; $h = $bmp.Height
$rect = New-Object System.Drawing.Rectangle 0, 0, $w, $h
$bd = $bmp.LockBits($rect, [System.Drawing.Imaging.ImageLockMode]::ReadOnly, [System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
$stride = $bd.Stride
$bytes = New-Object byte[] ($stride * $h)
[System.Runtime.InteropServices.Marshal]::Copy($bd.Scan0, $bytes, 0, $bytes.Length)
$bmp.UnlockBits($bd)
$bmp.Dispose()

function Lum([int]$x, [int]$y) {
    $i = $y * $stride + $x * 3
    return [math]::Round(0.114 * $bytes[$i] + 0.587 * $bytes[$i + 1] + 0.299 * $bytes[$i + 2], 1)
}

if ($X1 -lt 0) { $X1 = $(if ($Vertical) { $h - 1 } else { $w - 1 }) }

$line = @()
for ($v = $X0; $v -le $X1; $v++) {
    $l = if ($Vertical) { Lum $Y $v } else { Lum $v $Y }
    $line += ("{0}:{1}" -f $v, $l)
}
$axis = if ($Vertical) { "column x=$Y" } else { "row y=$Y" }
Write-Output "$axis  luminance (pos:lum)"
Write-Output ($line -join "  ")

# 50% crossings
Write-Output ""
Write-Output "50% crossings (lum 127.5):"
for ($v = $X0; $v -lt $X1; $v++) {
    $a = if ($Vertical) { Lum $Y $v } else { Lum $v $Y }
    $b = if ($Vertical) { Lum $Y ($v + 1) } else { Lum ($v + 1) $Y }
    if (($a - 127.5) * ($b - 127.5) -lt 0) {
        $t = (127.5 - $a) / ($b - $a)
        $pos = $v + $t
        $dir = if ($b -lt $a) { "white->BLACK" } else { "BLACK->white" }
        Write-Output ("  {0,8:N2}   {1}" -f $pos, $dir)
    }
}
