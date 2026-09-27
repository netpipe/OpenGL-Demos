param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$Out
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

$grey = New-Object byte[] ($w * $h)
for ($y = 0; $y -lt $h; $y++) {
    for ($x = 0; $x -lt $w; $x++) {
        $i = $y * $stride + $x * 3
        $lum = 0.114 * $bytes[$i] + 0.587 * $bytes[$i + 1] + 0.299 * $bytes[$i + 2]
        $grey[$y * $w + $x] = [byte][math]::Min(255, [math]::Max(0, [math]::Round($lum)))
    }
}

$fs = [System.IO.File]::Create($Out)
$hdr = [System.Text.Encoding]::ASCII.GetBytes("P5`n$w $h`n255`n")
$fs.Write($hdr, 0, $hdr.Length)
$fs.Write($grey, 0, $grey.Length)
$fs.Close()

Write-Output "wrote $Out ($w x $h)"
